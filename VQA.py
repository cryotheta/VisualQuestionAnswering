import torch
import torchvision.models as models
from torch import nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from data import AnswerTokenzier, load_image, load_data # These are expected to be in data.py
from transformers import BertTokenizer, BertModel # BertModel is not directly used by EncoderDecoder shown
from tqdm import tqdm
from tensorboardX import SummaryWriter
# from torch.utils.tensorboard import SummaryWriter # Alternative, sticking to original import
from torch.cuda.amp import GradScaler, autocast # Original had this commented out, keeping structure
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import argparse
from torch.nn import functional as F
from models import *
from utils import *
# All class definitions and helper functions live outside the main pipeline function.
def run_pipeline(args):
    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")        
    if args.mode != 'train':
        print(f"Warning: Mode '{args.mode}' is specified. This script is primarily set up for 'train' mode.")
        # For a 'test' mode, one would typically load a model and evaluate on the test set.
        # This basic structure will proceed assuming training, but specific test logic could be added.

    # Determine save path for checkpoints and logs
    if args.save_path:
        final_save_dir = args.save_path
    else:
        final_save_dir = 'checkpoints_vqa' # Default directory for checkpoints and logs
    os.makedirs(final_save_dir, exist_ok=True)
    
    runs_log_dir = os.path.join(final_save_dir, "runs")
    # os.makedirs(runs_log_dir, exist_ok=True) # SummaryWriter creates the directory
    writer = SummaryWriter(log_dir=runs_log_dir)

    # BERT Tokenizer for questions
    bert_tokenizer_q = BertTokenizer.from_pretrained("bert-base-uncased")
    
    # --- Data Loading and Preprocessing ---
    # args.dataset is the base path like /path/to/CLEVR_COL774_A4
    base_dataset_dir = args.dataset 
    
    train_json_path = os.path.join(base_dataset_dir, 'questions', 'CLEVR_trainA_questions.json')
    val_json_path = os.path.join(base_dataset_dir, 'questions', 'CLEVR_valA_questions.json')
    # test_json_path used 'valA' split for images in original for test questions, assuming this means test questions map to val images.
    test_json_path = os.path.join(base_dataset_dir, 'questions', 'CLEVR_testA_questions.json')

    print(f"Loading training data from: {train_json_path}")
    train_img_files, train_questions_raw, train_answers_raw = load_data(train_json_path)
    print(f"Loading validation data from: {val_json_path}")
    val_img_files, val_questions_raw, val_answers_raw = load_data(val_json_path)
    print(f"Loading test data from: {test_json_path}") # For setting up test_dataloader
    test_img_files, test_questions_raw, test_answers_raw = load_data(test_json_path)

    # Answer Tokenizer (specific to the dataset's answers)
    # Build vocabulary based on training answers
    answer_tokenizer_a = AnswerTokenzier(train_answers_raw)
    answer_tokenizer_a.build_vocab()
    num_answer_classes = answer_tokenizer_a.vocab_size
    print(f"Answer vocabulary size: {num_answer_classes}")

    # Datasets
    max_question_length = 64 # Max length for question token sequences

    train_dataset = ImageQADataset(train_img_files, train_questions_raw, train_answers_raw, 'trainA', 
                                   bert_tokenizer_q, answer_tokenizer_a, max_question_length)
    val_dataset = ImageQADataset(val_img_files, val_questions_raw, val_answers_raw, 'valA', 
                                 bert_tokenizer_q, answer_tokenizer_a, max_question_length)
    # Original code used 'valA' for images with test questions.
    test_dataset = ImageQADataset(test_img_files, test_questions_raw, test_answers_raw, 'testA', 
                                  bert_tokenizer_q, answer_tokenizer_a, max_question_length)

    # DataLoaders
    batch_size = 128  # From original script
    num_workers = 8 # From original script (adjust based on system capabilities)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, 
                                  num_workers=num_workers, prefetch_factor=2 if num_workers > 0 else None)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, 
                                num_workers=num_workers, prefetch_factor=2 if num_workers > 0 else None)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, 
                                 num_workers=num_workers, prefetch_factor=2 if num_workers > 0 else None)

    # --- Model Initialization ---
    # ResNet101 image encoder
     # Expected in current directory
    resnet_weights_file = "resnet101-63fe2227.pth"
    if os.path.exists(resnet_weights_file):
        resnet101_model = models.resnet101(weights=None) # Load custom weights
        print(f"Loading ResNet101 weights from: {resnet_weights_file}")
        resnet101_model.load_state_dict(torch.load(resnet_weights_file, map_location=device))
    else:
        resnet101_model = models.resnet101(pretrained=True)
        print(f"Warning: ResNet weights file '{resnet_weights_file}' not found. Using default initialized ResNet101.")
        # Consider using torchvision's pretrained weights as a fallback:
        # resnet101_model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        # print("Initialized ResNet101 with ImageNet pretrained weights from torchvision.")


    resnet101_features = remove_layer(resnet101_model, 2) # Remove classifier (avgpool, fc)
    # for param in resnet101_features.parameters(): # Freeze ResNet weights
        # param.requires_grad = False
    resnet101_features.to(device)
    
    # Text Encoder
    text_encoder_module = TextEncoder(d_embedding=768, vocab_size= max(bert_tokenizer_q.get_vocab().values()) , max_len=max_question_length)

    # Full VQA Model
    vqa_model = EncoderDecoder(
        d_embedding=768,
        resnet101=resnet101_features,
        text_encoder=text_encoder_module,
        num_classes=num_answer_classes
    ).to(device)

    if args.model_path:
        if os.path.exists(args.model_path):
            print(f"Loading pre-trained VQA model from: {args.model_path}")
            vqa_model.load_state_dict(torch.load(args.model_path, map_location=device))
        else:
            print(f"Warning: Specified model_path '{args.model_path}' not found. Training from scratch or ResNet backbone.")
    bert_model = BertModel.from_pretrained("bert-base-uncased")
    bert_embedding_weights = bert_model.embeddings.word_embeddings.weight.to(device)
    print(bert_embedding_weights.shape)
    vqa_model.text_encoder.embedding = nn.Embedding.from_pretrained(bert_embedding_weights, freeze=False)

    # --- Training Setup ---
    optimizer = torch.optim.Adam(vqa_model.parameters(), lr=1e-5, weight_decay=1e-5) # LR from final original version
    
    num_epochs = 40  # From original script
    warmup_steps = 10000 # From original script

    # Schedulers
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-8, # Start LR very small
        end_factor=1.0,    # Linearly increase to base LR
        total_iters=warmup_steps
    )
    # Plateau scheduler applied after warmup, based on training loss (as per original logic)
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    criterion = nn.CrossEntropyLoss()
    current_global_step = 0 # For TensorBoard logging

    # Automatic Mixed Precision (AMP) - original script had this commented out.
    # use_amp = torch.cuda.is_available() # Enable if desired, can speed up training on compatible GPUs
    # scaler = GradScaler(enabled=use_amp)

    # --- Training Loop (if mode is 'train') ---
    if args.mode == 'train':
        print("Starting training...")
        for epoch in range(num_epochs):
            vqa_model.train()
            epoch_train_loss = 0
            epoch_train_accuracy_samples = []

            train_pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
            for i, (batch_imgs, batch_q_ids, batch_ans_ids) in train_pbar:
                batch_imgs = batch_imgs.to(device)
                batch_q_ids = batch_q_ids.to(device)
                batch_ans_ids = batch_ans_ids.to(device)

                optimizer.zero_grad()

                # Original AMP logic (commented out):
                # with autocast(enabled=use_amp):
                #     outputs = vqa_model(batch_imgs, batch_q_ids)
                #     loss = criterion(outputs, batch_ans_ids)
                # scaler.scale(loss).backward()
                # scaler.step(optimizer)
                # scaler.update()
                
                outputs = vqa_model(batch_imgs, batch_q_ids)
               # loss = criterion(outputs, batch_ans_ids)
                loss = focal_loss(outputs, batch_ans_ids,alpha=None )
                
                loss.backward()
                optimizer.step()

                epoch_train_loss += loss.item()
                
                # Learning rate scheduling
                step_for_scheduler = epoch * len(train_dataloader) + i
                if step_for_scheduler < warmup_steps:
                    warmup_scheduler.step()
                
                # Logging batch metrics periodically
                if i % 100 == 0:
                    preds = torch.argmax(outputs, dim=-1)
                    batch_metrics = compute_metrics(preds, batch_ans_ids)
                    epoch_train_accuracy_samples.append(batch_metrics["accuracy"])

                    train_pbar.set_postfix({
                        'Loss': f"{loss.item():.4f}", 
                        'Acc': f"{batch_metrics['accuracy']:.4f}",
                        'LR': f"{optimizer.param_groups[0]['lr']:.2e}"
                    })

                    writer.add_scalar("Train/Batch_Loss", loss.item(), current_global_step)
                    writer.add_scalar("Train/Batch_Accuracy", batch_metrics["accuracy"], current_global_step)
                    writer.add_scalar("Train/Learning_Rate", optimizer.param_groups[0]['lr'], current_global_step)
                    # writer.add_scalar("Monitor/Precision", batch_metrics["precision"], current_global_step) # Optional more detailed logging
                    # writer.add_scalar("Monitor/Recall", batch_metrics["recall"], current_global_step)
                    # writer.add_scalar("Monitor/F1", batch_metrics["f1"], current_global_step)
                    current_global_step += 1
            
            avg_epoch_train_loss = epoch_train_loss / len(train_dataloader)
            writer.add_scalar("Train/Epoch_Avg_Loss", avg_epoch_train_loss, epoch)
            if epoch_train_accuracy_samples:
                writer.add_scalar("Train/Epoch_Avg_Sampled_Accuracy", np.mean(epoch_train_accuracy_samples), epoch)

            # Step the ReduceLROnPlateau scheduler using average training loss for the epoch
            # (Original did this after epoch > 1; here, apply if past warmup)
            

            # Save checkpoint
            checkpoint_filename = f'checkpoint_epoch_{epoch+1}.pth'
            checkpoint_path = os.path.join(final_save_dir, checkpoint_filename)
            torch.save(vqa_model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

            # --- Validation Loop (after each training epoch) ---
            vqa_model.eval()
            val_losses = []
            all_val_preds = []
            all_val_labels = []

            val_pbar = tqdm(val_dataloader, total=len(val_dataloader), desc=f"Epoch {epoch+1}/{num_epochs} [Validate]")
            with torch.no_grad():
                for batch_imgs, batch_q_ids, batch_ans_ids in val_pbar:
                    batch_imgs = batch_imgs.to(device)
                    batch_q_ids = batch_q_ids.to(device)
                    batch_ans_ids = batch_ans_ids.to(device)

                    outputs = vqa_model(batch_imgs, batch_q_ids)
                    loss = criterion(outputs, batch_ans_ids)
                    val_losses.append(loss.item())
                    
                    preds = torch.argmax(outputs, dim=-1)
                    all_val_preds.append(preds)
                    all_val_labels.append(batch_ans_ids)
            
            avg_val_loss = np.mean(val_losses)
            val_preds_tensor = torch.cat(all_val_preds)
            val_labels_tensor = torch.cat(all_val_labels)
            val_metrics = compute_metrics(val_preds_tensor, val_labels_tensor)

            writer.add_scalar("Val/Epoch_Avg_Loss", avg_val_loss, epoch)
            writer.add_scalar("Val/Epoch_Accuracy", val_metrics["accuracy"], epoch)
            writer.add_scalar("Val/Epoch_Precision", val_metrics["precision"], epoch)
            writer.add_scalar("Val/Epoch_Recall", val_metrics["recall"], epoch)
            writer.add_scalar("Val/Epoch_F1", val_metrics["f1"], epoch)

            print(f"Epoch {epoch+1} Validation: Avg Loss: {avg_val_loss:.4f}, Accuracy: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}")
            if step_for_scheduler >= warmup_steps: # Ensure warmup is complete
                 plateau_scheduler.step(avg_val_loss)
        writer.close()
        print("Training finished.")
    
    # elif args.mode == 'test':

        # Implement test logic here: load model (likely from args.model_path or best checkpoint),
        # run evaluation on test_dataloader, print/save results.
        # print("Test mode execution...")
        # ... (load model if not already loaded, model.eval(), loop through test_dataloader, compute metrics)
def run_inference_for_eval(model, dataloader, criterion, device, dataloader_name_desc):
    model.eval() # Ensure model is in eval mode
    total_loss_sum = 0.0
    all_predictions_list = []
    all_ground_truths_list = []

    progress_bar = tqdm(dataloader, desc=f"Inferring on {dataloader_name_desc}", leave=False)
    with torch.no_grad():
        for batch_idx, (batch_images, batch_question_ids, batch_answer_ids) in enumerate(progress_bar):
            batch_images = batch_images.to(device)
            batch_question_ids = batch_question_ids.to(device)
            batch_answer_ids = batch_answer_ids.to(device)

            outputs = model(batch_images, batch_question_ids)
            loss = criterion(outputs, batch_answer_ids)
            # Accumulate total loss correctly, considering varying batch sizes (last batch)
            total_loss_sum += loss.item() * batch_images.size(0)

            predicted_labels = torch.argmax(outputs, dim=-1)
            all_predictions_list.append(predicted_labels.cpu())
            all_ground_truths_list.append(batch_answer_ids.cpu())

    # Calculate average loss over the entire dataset
    avg_loss_val = total_loss_sum / len(dataloader.dataset) if len(dataloader.dataset) > 0 else 0.0
    
    all_predictions_tensor = torch.cat(all_predictions_list)
    all_ground_truths_tensor = torch.cat(all_ground_truths_list)
    
    calculated_metrics = compute_metrics(all_predictions_tensor, all_ground_truths_tensor)
    
    return avg_loss_val, calculated_metrics



def eval_pipeline(args):
    print("Starting evaluation pipeline...")
    if not args.model_path or not os.path.exists(args.model_path):
        print(f"Error: --model_path '{args.model_path}' is required for evaluation and must exist.")
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # BERT Tokenizer for questions
    bert_tokenizer_q = BertTokenizer.from_pretrained("bert-base-uncased")
    max_question_length = 64 # Should be consistent with training

    base_dataset_dir = args.dataset
    train_json_path_for_ans_tok = os.path.join(base_dataset_dir, 'questions', 'CLEVR_trainA_questions.json')
    if not os.path.exists(train_json_path_for_ans_tok):
        print(f"Error: Training questions JSON file not found at {train_json_path_for_ans_tok} for AnswerTokenizer setup.")
        return
    _, _, train_answers_raw_for_ans_tok = load_data(train_json_path_for_ans_tok)
    
    answer_tokenizer_a = AnswerTokenzier(train_answers_raw_for_ans_tok)
    answer_tokenizer_a.build_vocab()
    num_answer_classes = answer_tokenizer_a.vocab_size
    print(f"Answer vocabulary size (derived from training data): {num_answer_classes}")

    # --- Model Component Initialization (ResNet, TextEncoder) ---
    # ResNet101 image encoder
     # Load custom weights if specified
    resnet_weights_file = "resnet101-63fe2227.pth" # Path relative to script or absolute
                                                 # This should ideally be a configurable path or handled more robustly.
    if os.path.exists(resnet_weights_file):
        resnet101_model = models.resnet101(weights=None)
        print(f"Loading ResNet101 base weights from: {resnet_weights_file}")
        resnet101_model.load_state_dict(torch.load(resnet_weights_file, map_location=device))
    else:
        resnet101_model = models.resnet101(pretrained=True)
        # This warning is important if the model was trained with these specific ResNet weights
        print(f"Warning: Pre-trained ResNet weights file '{resnet_weights_file}' not found. Using randomly initialized/downloaded ResNet101.")
        # Alternatively, load torchvision's ImageNet weights:
        # resnet101_model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        # print("Initialized ResNet101 with ImageNet pretrained weights from torchvision.")
        
    resnet101_features = remove_layer(resnet101_model, 2) # Remove classifier (avgpool, fc)
    resnet101_features.to(device) # Move to specified device
    text_encoder_module = TextEncoder(
        d_embedding=768, # Standard BERT base embedding size
        vocab_size=bert_tokenizer_q.vocab_size , 
        max_len=max_question_length
    ).to(device) # Move to specified device
    vqa_model = EncoderDecoder(
        d_embedding=768,
        resnet101=resnet101_features,
        text_encoder=text_encoder_module,
        num_classes=num_answer_classes
    ).to(device)

    print(f"Loading VQA model checkpoint from: {args.model_path}")
    vqa_model.load_state_dict(torch.load(args.model_path, map_location=device))
    vqa_model.eval() # Set model to evaluation mode explicitly

    # --- Data Loading and Dataloaders for Evaluation ---
    # We can evaluate on validation, test, and optionally on the training set.
    datasets_to_evaluate_configs = {
        "Validation": {
            "json_file": "CLEVR_valA_questions.json", 
            "image_split": "valA" # Image folder split for validation questions
        },
        "Test": {
            "json_file": "CLEVR_testA_questions.json", 
            "image_split": "testA" # Original script used 'valA' images for test questions
        },
        "TestB": {
            "json_file": "CLEVR_testB_questions.json", 
            "image_split": "testB" # Original script used 'valA' images for test questions
        },
        "Train (for eval comparison)": {
            "json_file": "CLEVR_trainA_questions.json",
            "image_split": "trainA" # Image folder split for training questions
        }
    }

    dataloaders_for_eval = {}
    for name, config in datasets_to_evaluate_configs.items():
        json_path = os.path.join(base_dataset_dir, 'questions', config["json_file"])
        if os.path.exists(json_path):
            img_files, questions_raw, answers_raw = load_data(json_path)
            current_dataset = ImageQADataset(
                img_files, questions_raw, answers_raw, 
                config["image_split"], 
                bert_tokenizer_q, answer_tokenizer_a, max_question_length
            )
            num_workers=1
            dataloaders_for_eval[name] = DataLoader(
                current_dataset, 
                batch_size=128, # Use batch_size from args
                shuffle=False, # No shuffling for evaluation
                pin_memory=True, 
                num_workers=2, # Use num_workers from args
                prefetch_factor=2 if num_workers > 0 else None
            )
            print(f"Prepared {name} dataloader with {len(current_dataset)} samples.")
        else:
            print(f"Warning: {name} JSON file not found at {json_path}. Skipping {name} set evaluation.")
            
    if not dataloaders_for_eval:
        print("No datasets could be loaded for evaluation. Exiting.")
        return

    criterion = nn.CrossEntropyLoss() # For calculating loss, if desired during evaluation

    # --- Run Inference and Compute Metrics for each Dataloader ---
    evaluation_results_summary = {}

    for name, dataloader in dataloaders_for_eval.items():
        print(f"\n--- Evaluating on {name} set ---")
        avg_loss, metrics = run_inference_for_eval(vqa_model, dataloader, criterion, device, name)
        evaluation_results_summary[name] = {"loss": avg_loss, "metrics": metrics}
        print(evaluation_results_summary[name])

    # --- Print All Metrics in Pretty Format ---
    print("\n\n--- Overall Evaluation Summary ---")
    header = f"{'Dataset':<30} | {'Avg Loss':<10} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-score':<10}"
    print(header)
    print("-" * len(header))
    for name, results in evaluation_results_summary.items():
        loss_str = f"{results['loss']:.4f}"
        acc_str = f"{results['metrics']['accuracy']:.4f}"
        prec_str = f"{results['metrics']['precision']:.4f}"
        rec_str = f"{results['metrics']['recall']:.4f}"
        f1_str = f"{results['metrics']['f1']:.4f}"
        print(f"{name:<30} | {loss_str:<10} | {acc_str:<10} | {prec_str:<10} | {rec_str:<10} | {f1_str:<10}")
    
    print("\nEvaluation finished.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train or test a VQA model.")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'inference'], 
                        help="Mode of operation: 'train' for training, 'inference' for evaluation.")
    parser.add_argument('--dataset', type=str, required=True, 
                        help="Path to the base dataset directory (e.g., where CLEVR_COL774_A4/questions/ is located).")
    parser.add_argument('--save_path', type=str, default=None, 
                        help="Optional: Directory to save checkpoints and TensorBoard logs. Defaults to './checkpoints_vqa'.")
    parser.add_argument('--model_path', type=str, default=None, 
                        help="Optional: Path to a pre-trained model checkpoint to load (for resuming training or for testing).")

    #python3 refactored_10_f.py --mode 'train' --dataset '.' --save_path './save_path' --model_path 'checkpoint_r7/Checkpoint3.pth' 
    
    # Additional arguments could be added here (e.g., batch_size, lr, epochs)
    # For example:
    # parser.add_argument('--batch_size', type=int, default=128, help="Batch size for DataLoaders.")
    # parser.add_argument('--epochs', type=int, default=40, help="Number of training epochs.")
    # parser.add_argument('--lr', type=float, default=1e-5, help="Initial learning rate for Adam optimizer.")

    cli_args = parser.parse_args()
    if cli_args.mode=='train':
        run_pipeline(cli_args)
    elif cli_args.mode=='inference':
        eval_pipeline(cli_args)
    else:
        print("Invalid argument for mode should be either train or inference")