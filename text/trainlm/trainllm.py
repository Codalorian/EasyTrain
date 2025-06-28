import torch
import os
import logging
import json
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from datasets import load_dataset, Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    set_seed
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
    TaskType
)
from accelerate import Accelerator
from accelerate.logging import get_logger

# Set up logging
logger = get_logger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

@dataclass
class TrainingConfig:
    model_name: str = field(metadata={"help": "Name or path to pre-trained model"})
    tokenizer_name: Optional[str] = field(default="AutoTokenizer", metadata={"help": "Name or path to tokenizer"})
    dataset_name: str = field(metadata={"help": "Name or path to dataset (json, jsonl, csv, txt)"})
    batch_size: int = field(default=8, metadata={"help": "Training batch size"})
    training_steps: int = field(default=400, metadata={"help": "Total training steps (minimum 400)"})
    learning_rate: float = field(default=4e-5, metadata={"help": "Learning rate"})
    logging_dir: str = field(default="./logs", metadata={"help": "Directory for logs"})
    checkpoint_interval: int = field(default=10, metadata={"help": "Save checkpoint every N steps"})
    output_dir: str = field(default="./output", metadata={"help": "Output directory for model"})
    use_accelerate: bool = field(default=False, metadata={"help": "Use Hugging Face Accelerate"})
    use_peft: bool = field(default=False, metadata={"help": "Use Parameter Efficient Fine-Tuning (PEFT)"})
    lora_rank: int = field(default=8, metadata={"help": "LoRA attention dimension"})
    lora_alpha: int = field(default=16, metadata={"help": "LoRA alpha scaling parameter"})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout rate"})
    validation_split: float = field(default=0.1, metadata={"help": "Validation dataset split"})
    max_seq_length: int = field(default=512, metadata={"help": "Maximum sequence length"})
    seed: int = field(default=42, metadata={"help": "Random seed"})

class DatasetHandler:
    """Handles dataset loading and column detection"""
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.file_type = self._detect_file_type()
        self.columns = self._detect_columns()
    
    def _detect_file_type(self) -> str:
        """Detect file type from extension"""
        ext = os.path.splitext(self.dataset_path)[-1].lower()
        if ext in ['.json', '.jsonl']:
            return 'json'
        elif ext == '.csv':
            return 'csv'
        elif ext == '.txt':
            return 'txt'
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    
    def _detect_columns(self) -> Dict[str, str]:
        """Detect dataset columns/headers"""
        if self.file_type == 'json':
            with open(self.dataset_path, 'r') as f:
                sample = json.loads(f.readline())
            return {k: type(v).__name__ for k, v in sample.items()}
        elif self.file_type == 'csv':
            import pandas as pd
            df = pd.read_csv(self.dataset_path, nrows=1)
            return {col: str(dtype) for col, dtype in df.dtypes.items()}
        return {}  # No columns for txt
    
    def load_dataset(self) -> DatasetDict:
        """Load dataset with appropriate format handling"""
        if self.file_type == 'json':
            dataset = load_dataset('json', data_files=self.dataset_path)
        elif self.file_type == 'csv':
            dataset = load_dataset('csv', data_files=self.dataset_path)
        elif self.file_type == 'txt':
            with open(self.dataset_path, 'r') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
            dataset = Dataset.from_dict({"text": lines})
        else:
            raise ValueError(f"Unsupported file type: {self.file_type}")
        
        # Split dataset
        return dataset.train_test_split(test_size=0.1, seed=42)

class ModelTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        set_seed(config.seed)
        
        # Initialize accelerator
        self.accelerator = Accelerator() if config.use_accelerate else None
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer_name or config.model_name,
            use_fast=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(config.model_name)
        
        # Apply PEFT if enabled
        if config.use_peft:
            self.model = self._apply_peft()
        
        # Resize token embeddings
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Load dataset
        self.dataset_handler = DatasetHandler(config.dataset_name)
        self.dataset = self.dataset_handler.load_dataset()
        
        # Tokenize dataset
        self.tokenized_datasets = self._tokenize_dataset()
        
        # Set up training arguments
        self.training_args = TrainingArguments(
            output_dir=config.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=1000,  # We'll control steps manually
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            max_steps=config.training_steps,
            logging_dir=config.logging_dir,
            logging_steps=10,
            save_steps=config.checkpoint_interval,
            save_total_limit=3,
            remove_unused_columns=False,
            report_to="tensorboard",
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            push_to_hub=False,
            fp16=torch.cuda.is_available(),
            gradient_checkpointing=True,
            seed=config.seed
        )
        
        # Create data collator
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
    
    def _apply_peft(self):
        """Apply PEFT configuration to the model"""
        lora_config = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        # For 8-bit quantization support
        if "int8" in self.config.model_name.lower():
            model = prepare_model_for_int8_training(self.model)
        
        return get_peft_model(self.model, lora_config)
    
    def _tokenize_dataset(self) -> DatasetDict:
        """Tokenize the dataset"""
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.config.max_seq_length,
                padding="max_length",
                return_special_tokens_mask=True,
            )
        
        return self.dataset.map(
            tokenize_function,
            batched=True,
            num_proc=4,
            remove_columns=["text"],
            desc="Tokenizing dataset"
        )
    
    def train(self):
        """Run the training process"""
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.tokenized_datasets["train"],
            eval_dataset=self.tokenized_datasets["test"],
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
        )
        
        # Start training
        logger.info("Starting training...")
        trainer.train()
        
        # Save final model
        trainer.save_model(self.config.output_dir)
        self.tokenizer.save_pretrained(self.config.output_dir)
        
        # Save PEFT config if applicable
        if self.config.use_peft:
            peft_config_path = os.path.join(self.config.output_dir, "peft_config.json")
            with open(peft_config_path, "w") as f:
                json.dump(self.config.__dict__, f, indent=2)
        
        logger.info(f"Training completed. Model saved to {self.config.output_dir}")
        return trainer

def run_training(config: TrainingConfig):
    """Main function to run training with specified configuration"""
    logger.info("Initializing model trainer...")
    trainer = ModelTrainer(config)
    
    logger.info("Starting training process...")
    trainer.train()
