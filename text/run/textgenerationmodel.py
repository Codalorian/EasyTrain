import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, TextStreamer


class TextGenerationModel:
    def __init__(
        self,
        model_name,
        system_prompt="You are an AI assistant",
        temperature=0.7,
        use_accelerate=False,
        use_deepspeed=False
    ):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)

        self.system_prompt = system_prompt
        self.temperature = temperature

        # Future: integrate Accelerate or DeepSpeed here
        if use_accelerate:
            raise NotImplementedError("Accelerate integration is not yet supported.")
        if use_deepspeed:
            raise NotImplementedError("Deepspeed integration is not yet supported.")

    def generate_response(self, prompt, max_new_tokens=200, stream=False):
        """
        Generate a response to a given prompt using the loaded model.

        Args:
            prompt (str): User input to respond to.
            max_new_tokens (int): Maximum number of tokens to generate.
            stream (bool): Whether to stream tokens in real-time.

        Returns:
            str: Generated response text.
        """
        input_text = f"{self.system_prompt}\n\n{prompt}"
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(self.device)

        generation_config = GenerationConfig(
            temperature=self.temperature,
            top_p=0.9,
            do_sample=True,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id
        )

        if stream:
            streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
            outputs = self.model.generate(
                **inputs,
                generation_config=generation_config,
                streamer=streamer
            )
            # Convert streamer output to string
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            outputs = self.model.generate(
                **inputs,
                generation_config=generation_config
            )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
