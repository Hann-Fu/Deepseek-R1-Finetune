"""
References:
- Model: https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B
- Dataset: https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT
"""

# * ====================================
# * IMPORTS AND SETUP
# * ====================================

import os
import torch
import warnings
import wandb  # For experiment tracking and logging
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    pipeline,
    logging,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig  # For parameter-efficient fine-tuning using LoRA
from trl import SFTTrainer  # Specialized trainer for supervised fine-tuning (SFT)

# * -------------------------------
# * Warning suppression and logging settings
# * -------------------------------
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.set_verbosity_error()  # Hide non-critical logs to keep output clean

# -------------------------------
# Initialize Weights & Biases (wandb) for experiment tracking.
# This helps in logging training metrics, hyperparameters, and debugging.
# -------------------------------
wandb.init(
    project="deepseek-r1-medical-finetuning",
    config={
        "model_name": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",  # ! Model name for loading pre-trained weights
        "learning_rate": 5e-5,  # ! Initial learning rate for fine-tuning
        "batch_size": 1,  # ! Batch size for training
        "num_epochs": 2,  # ! Number of epochs for training
        "hardware": "Nvidia RTX 3080Ti",  
        "dataset": "medical-o1-reasoning-SFT",  # ! Dataset used for training
        "lora_rank": 8,  # ! LoRA rank for parameter-efficient tuning
        "lora_alpha": 16  # ! LoRA alpha parameter
    }
)

# -------------------------------
# Device Setup: Determine the best device available.
# -------------------------------

if torch.cuda.is_available():
    print("Using CUDA GPU")
    device = torch.device("cuda")  # ! Use GPU for faster computation
else:
    print("Using CPU")
    device = torch.device("cpu")  # ! Fallback to CPU if GPU is unavailable


# * ====================================
# * DATA PREPARATION
# * ====================================

def prepare_dataset(tokenizer):
    """
    Prepares the medical reasoning dataset for training.
    
    Why: The model requires prompts in a specific format to encourage
    step-by-step reasoning. This function formats the raw data accordingly.
    
    How: 
      - Loads the dataset from Hugging Face.
      - Formats each example with a custom prompt including the question, 
        chain-of-thought (CoT) reasoning, and final answer.
      - Uses a subset (5% of training data) to keep the tutorial run fast.
      - Tokenizes the text with settings optimized for speed.
    
    When to modify:
      - Adjust the train_size and test_size parameters for a full run.
      - Change max_length if your use case requires longer sequences.
      - Update the prompt format to suit different tasks or datasets.
    """
    # Load dataset; Edit the dataset name to load different datasets.
    dataset = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", "en")  # ! Load the specific dataset for training
    print(f"Dataset loaded with {len(dataset['train'])} training examples")
    
    def format_instruction(sample):
        # Recommended prompt format for better reasoning performance(Check the R1 official website for more information).
        return f"""Please reason step by step:

                Question: {sample['Question']}

                Let's solve this step by step:
                {sample['Complex_CoT']}

                Final Answer: {sample['Response']}"""
    
    # Subsample the dataset for faster iterations during the tutorial.
    dataset = dataset["train"].train_test_split(train_size=0.05, test_size=0.01, seed=42)  # ! Use a small subset for quick experimentation
    
    # Format each training example.
    train_dataset = dataset["train"].map(
        lambda x: {"text": format_instruction(x)},
        remove_columns=dataset["train"].column_names,
        num_proc=os.cpu_count()  # Parallel processing using available CPU cores.
    )
    
    # Tokenize the formatted text.
    # Note: max_length is set to 1024 tokens for a balance between speed and context length.
    train_dataset = train_dataset.map(
        lambda x: tokenizer(
            x["text"],
            truncation=True,
            padding="max_length",
            max_length=1024,  # ! Maximum token length for each input sequence
            return_tensors=None,
        ),
        remove_columns=["text"],
        num_proc=os.cpu_count()
    )
    
    print(f"\nUsing {len(train_dataset)} examples for training")
    print("\nSample formatted data:")
    print(format_instruction(dataset["train"][0]))
    
    return train_dataset

# ! ====================================
# ! MODEL SETUP
# ! ====================================

def setup_model():
    """
    Sets up the DeepSeek model with configurations tailored for Apple Silicon.
    
    Why: Loading the model with specific settings such as:
         - Memory optimizations (disabling KV-cache).
         - Reduced precision settings (float16) to conserve memory.
    
    How:
         - Loads the tokenizer and sets padding tokens.
         - Loads the model using the 'from_pretrained' method with custom arguments.
    
    When to modify:
         - Change torch_dtype if you want to experiment with different precision.
         - Adjust max_memory if your hardware configuration is different.
         - Modify use_cache if caching is needed for inference.
    """
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"  # ! Model identifier for loading pre-trained weights
    
    # Load the tokenizer and ensure proper padding settings.
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" 
    
    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",  # ! Automatically map model to available devices
        trust_remote_code=True,
        torch_dtype=torch.float16,  # ! Use float16 precision for reduced memory usage
        use_cache=False,  # ! Disable KV-cache to save memory during training
        max_memory={0: "12GB"},  # ! Set maximum memory allocation for device 0
    )
    
    # Memory optimization: If available, enable input gradients for efficiency.
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    
    return model, tokenizer

# ! ====================================
# ! TRAINER SETUP WITH LoRA CONFIGURATION
# ! ====================================

def setup_trainer(model, tokenizer, train_dataset, eval_dataset):
    """
    ! Sets up the SFTTrainer for LoRA-based fine-tuning.
    
    Why:
         - LoRA (Low-Rank Adaptation) allows parameter-efficient fine-tuning by only training a subset of parameters.
         - This is particularly useful for large models on limited hardware.
         - TrainingArguments are customized for faster iterations on Apple Silicon.
    
    How:
         - Configures LoRA with specified alpha, dropout, rank (r), and target modules.
         - Sets training hyperparameters like learning rate, batch size, gradient accumulation, and checkpointing.
    
    When to modify:
         - Adjust peft_config parameters (e.g., lora_alpha, r) to trade off between performance and resource usage.
         - Change training_args (e.g., num_train_epochs, per_device_train_batch_size) based on your dataset size and compute.
         - Enable or disable mixed precision (fp16/bf16) based on your device capabilities.
    """
    # Configure LoRA for efficient fine-tuning.
    peft_config = LoraConfig(
        lora_alpha=16,  # ! LoRA alpha parameter for controlling adaptation strength
        lora_dropout=0.1,  # ! Dropout rate for LoRA layers
        r=4,  # ! LoRA rank, lower for less computation, higher for more expressiveness
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"]  # ! Target modules for LoRA adaptation
    )
    
    # Set training arguments for the trainer.
    training_args = TrainingArguments(
        output_dir="deepseek-r1-medical-finetuning",  # Directory to save model checkpoints and outputs
        num_train_epochs=1,  # ! Number of full passes through the dataset (increase for real training)
        per_device_train_batch_size=2,  # ! Batch size per GPU/TPU core/CPU (adjust based on GPU memory)
        gradient_accumulation_steps=4,  # ! Accumulate gradients across N steps to simulate larger batches
        learning_rate=1e-4,  # ! Initial learning rate for AdamW optimizer (common starting point for fine-tuning)
        weight_decay=0.01,  # L2 regularization strength to prevent overfitting
        warmup_ratio=0.03,  # Percentage of training steps for linear learning rate warmup
        logging_steps=10,  # Log training metrics every X steps
        save_strategy="epoch",  # Save checkpoint at the end of each epoch
        save_total_limit=1,  # Maximum number of checkpoints to keep (prevents storage bloat)
        fp16=True,  # ! Use mixed precision training (requires compatible hardware)
        bf16=False,  # Use bfloat16 precision (alternative to fp16, requires Ampere+ GPU)
        optim="adamw_torch_fused",  # Optimizer implementation with fused operations for speed
        report_to="wandb",  # Integration with Weights & Biases for experiment tracking
        gradient_checkpointing=True,  # Trade compute for memory by recomputing activations
        group_by_length=True,  # Group similar-length sequences for more efficient processing
        max_grad_norm=0.3,  # Gradient clipping threshold to prevent exploding gradients
        dataloader_num_workers=0,  # Number of subprocesses for data loading (0=main process)
        remove_unused_columns=True,  # Strip unused columns from dataset to save memory
        run_name="deepseek-medical-tutorial",  # Display name for wandb tracking
        deepspeed=None,  # Configuration for DeepSpeed optimization (not used here)
        local_rank=-1,  # Local process rank for distributed training (-1 = not distributed)
        ddp_find_unused_parameters=None,  # Handling unused params in DistributedDataParallel
        torch_compile=False,  # Enable PyTorch 2.0's compiler for faster training (beta)
    )
    
    # Data collator: Handles dynamic padding for language modeling tasks.
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False  # Causal LM does not use masked language modeling.
    )
    
    # Initialize the SFTTrainer with our model, dataset, and training configurations.
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        peft_config=peft_config,
        args=training_args,
        data_collator=data_collator,
        processing_class=None  # SFTTrainer will handle input processing.
    )
    
    return trainer

# * ====================================
# * MODEL TESTING AFTER FINE-TUNING
# * ====================================

def test_model(model_path):
    """
    Tests the fine-tuned model by generating a response for a sample medical reasoning problem.
    
    Why:
         - To validate that the fine-tuning process succeeded and the model performs as expected.
         - Uses offloading to manage memory during inference on Apple Silicon.
    
    How:
         - Loads the fine-tuned model and tokenizer with offloading settings.
         - Sets up a text-generation pipeline with recommended parameters.
         - Generates a response for a predefined test case and logs the output.
    
    When to modify:
         - Change the 'test_problem' string to try different medical reasoning cases.
         - Adjust pipeline parameters (e.g., max_new_tokens, temperature) to experiment with generation behavior.
    """
    # Create an offload directory to manage memory during testing.
    os.makedirs("offload", exist_ok=True)
    
    # Load the model from the saved fine-tuned checkpoint.
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",  # ! Automatically map model to available devices
        trust_remote_code=True,
        torch_dtype=torch.float16,  # ! Use float16 precision for reduced memory usage
        offload_folder="offload",  # Specify offloading directory to handle memory constraints.
        offload_state_dict=True,
        use_cache=False,  # ! Disable KV-cache to save memory during inference
        max_memory={0: "24GB"},  # ! Set maximum memory allocation for device 0
    )
    
    # Load the tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="right"
    )
    
    # Setup a text-generation pipeline with parameters tuned for medical reasoning.
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",  # ! Automatically map model to available devices
        max_new_tokens=512,  # ! Maximum number of new tokens to generate
        temperature=0.6,  # ! Controls randomness in output (lower value = more deterministic).
        top_p=0.95,  # ! Nucleus sampling to limit the token pool.
        repetition_penalty=1.15,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Define a test prompt with the recommended "step-by-step" reasoning format.
    test_problem = """
                      Please reason step by step:
                      A 45-year-old patient presents with sudden onset chest pain, shortness of breath, and anxiety.
                      The pain is described as sharp and worsens with deep breathing. What is the most likely diagnosis 
                      and what immediate tests should be ordered?
                   """
    
    try:
        result = pipe(
            test_problem,
            max_new_tokens=512,
            temperature=0.6,
            top_p=0.95,
            repetition_penalty=1.15
        )
        
        print("\nTest Problem:", test_problem)
        print("\nModel Response:", result[0]["generated_text"])
        
        # Log test results to wandb for later review.
        wandb.log({
            "test_example": wandb.Table(
                columns=["Test Case", "Model Response"],
                data=[[test_problem, result[0]["generated_text"]]]
            )
        })
    except Exception as e:
        print(f"\nError during testing: {str(e)}")
        print("Model was saved successfully but testing failed. You can load the model separately for testing.")
    finally:
        # Clean up the offload folder to free disk space.
        if os.path.exists("offload"):
            import shutil
            shutil.rmtree("offload")

# * ====================================
# * MAIN FUNCTION: ORCHESTRATES THE WHOLE PROCESS
# * ====================================

def main():
    """
    * Main function to execute the complete fine-tuning workflow.
    
    Workflow:
      1. Set up the model and tokenizer.
      2. Prepare the training dataset.
      3. Configure and initialize the trainer with LoRA settings.
      4. Run training.
      5. Save the fine-tuned model.
      6. Test the model with a sample prompt.
      7. Finalize logging with wandb.
    
    Practical Tips:
      - Wrap critical sections in try/finally blocks to ensure resources (like wandb sessions) are cleaned up.
      - Modify the number of epochs and batch sizes based on available hardware.
      - Monitor training progress via wandb dashboard.
    """
    try:
        print("\nSetting up model...")
        model, tokenizer = setup_model()
        
        print("\nPreparing dataset...")
        train_dataset = prepare_dataset(tokenizer)
        
        print("\nSetting up trainer...")
        trainer = setup_trainer(model, tokenizer, train_dataset, None)
        
        print("\nStarting training...")
        trainer.train()  # ! This is where the fine-tuning happens.
        
        print("\nSaving model...")
        trainer.model.save_pretrained("./fine_tuned_model")  # ! Save the fine-tuned model for later use
        
        print("\nTesting model...")
        test_model("./fine_tuned_model")
        
    finally:
        wandb.finish()  # Ensure wandb logging is properly closed even if errors occur.

if __name__ == "__main__":
    main()
