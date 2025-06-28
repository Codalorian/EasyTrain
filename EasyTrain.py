import argparse
from text.trainlm import trainllm  # Import the training module
from text import textgenerationmodel  # Import the generation module

def main():
    parser = argparse.ArgumentParser(description="EasyTrain: LLM Training and Inference")
    subparsers = parser.add_subparsers(dest='command', required=True)

    # RunLM subparser
    runlm_parser = subparsers.add_parser('runlm', help='Run language model')
    runlm_parser.add_argument('--model', type=str, required=True, help='Model name or path')
    runlm_parser.add_argument('--prompt', type=str, required=True, help='User input prompt')
    runlm_parser.add_argument('--system-prompt', type=str, default="You are an AI assistant", help='System prompt')
    runlm_parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature')
    runlm_parser.add_argument('--use-accelerate', action='store_true', default=False, help='Use HuggingFace Accelerate')
    runlm_parser.add_argument('--use-deepspeed', action='store_true', default=False, help='Use DeepSpeed')
    runlm_parser.add_argument('--stream', action='store_true', default=False, help='Stream tokens in real-time')
    runlm_parser.add_argument('--max-new-tokens', type=int, default=200, help='Maximum tokens to generate')

    # TrainLM subparser
    trainlm_parser = subparsers.add_parser('trainlm', help='Train language model')
    trainlm_parser.add_argument('--model', type=str, required=True, help='Base model name or path')
    trainlm_parser.add_argument('--tokenizer', type=str, default="AutoTokenizer", help='Tokenizer name or path')
    trainlm_parser.add_argument('--dataset', type=str, required=True, help='Dataset path (json, jsonl, csv, txt)')
    trainlm_parser.add_argument('--batch-size', type=int, default=8, help='Training batch size')
    trainlm_parser.add_argument('--training-steps', type=int, default=400, help='Total training steps (min 400)')
    trainlm_parser.add_argument('--learning-rate', type=float, default=4e-5, help='Learning rate')
    trainlm_parser.add_argument('--logging-dir', type=str, default="./logs", help='Logging directory')
    trainlm_parser.add_argument('--checkpoints', type=int, default=10, help='Checkpoint interval')
    trainlm_parser.add_argument('--use-accelerate', action='store_true', default=False, help='Use HuggingFace Accelerate')
    trainlm_parser.add_argument('--use-peft', action='store_true', default=False, help='Use PEFT (LoRA) for efficient training')
    trainlm_parser.add_argument('--lora-rank', type=int, default=8, help='LoRA attention dimension (rank)')
    trainlm_parser.add_argument('--lora-alpha', type=int, default=16, help='LoRA alpha scaling parameter')
    trainlm_parser.add_argument('--lora-dropout', type=float, default=0.05, help='LoRA dropout rate')
    trainlm_parser.add_argument('--output-dir', type=str, default="./output", help='Output directory')

    args = parser.parse_args()

    if args.command == 'runlm':
        model = textgenerationmodel.TextGenerationModel(
            model_name=args.model,
            system_prompt=args.system_prompt,
            temperature=args.temperature,
            use_accelerate=args.use_accelerate,
            use_deepspeed=args.use_deepspeed
        )

        print("\n--- Response ---\n")
        response = model.generate_response(
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            stream=args.stream
        )
        print("\n----------------\n")

    elif args.command == 'trainlm':
        # Validate training steps
        if args.training_steps < 400:
            raise ValueError("Training steps must be at least 400")

        # Create training config
        config = trainllm.TrainingConfig(
            model_name=args.model,
            tokenizer_name=args.tokenizer,
            dataset_name=args.dataset,
            batch_size=args.batch_size,
            training_steps=args.training_steps,
            learning_rate=args.learning_rate,
            logging_dir=args.logging_dir,
            checkpoint_interval=args.checkpoints,
            output_dir=args.output_dir,
            use_accelerate=args.use_accelerate,
            use_peft=args.use_peft,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout
        )

        # Start training
        trainllm.run_training(config)

if __name__ == "__main__":
    main()