"""
pip3 install accelerate
pip3 install git+https://github.com/huggingface/transformers.git@main
pip3 show transformers
pip3 install --upgrade Pillow
pip3 install git+https://github.com/huggingface/trl.git
pip3 install deepspeed
pip3 install --upgrade jinja2
pip3 install wandb
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
git clone https://github.com/hu-po/trl.git
huggingface-cli login
wandb login
accelerate launch \
    examples/scripts/sft_vlm_llama.py \
    --model_name_or_path meta-llama/Llama-3.2-11B-Vision-Instruct \
    --dataset_name hu-po/rings-10k \
    --learning_rate 1e-5 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --output_dir sft-llama3.2-11b \
    --bf16 \
    --logging_steps 10 \
    --torch_dtype bfloat16 \
    --gradient_checkpointing
"""

import torch
from datasets import load_dataset
from transformers import AutoModelForVision2Seq, AutoProcessor, LlavaForConditionalGeneration
from PIL import Image

from trl import (
    ModelConfig,
    SFTConfig,
    SFTScriptArguments,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

if __name__ == "__main__":
    parser = TrlParser((SFTScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_and_config()
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    training_args.dataset_text_field = ""  # need a dummy field
    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}

    ################
    # Model, Tokenizer & Processor
    ################
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    processor = AutoProcessor.from_pretrained(
        model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code
    )

    model = AutoModelForVision2Seq.from_pretrained(
        model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code, **model_kwargs
    )

    ################
    # Create a data collator to encode text and image pairs
    ################
    def collate_fn(examples):
        # Modify the messages to match the expected format
        for idx, example in enumerate(examples):
            if idx < 3:  # Only print the first 3 examples
                print(f"\nExample {idx} before modification:")
                print("messages:", example["messages"])
                print("images:", example["images"])

            messages = []
            content_list = example["messages"]["content"]
            role_list = example["messages"]["role"]
            for i in range(len(role_list)):
                message = {
                    "role": role_list[i],
                    "content": []
                }
                content_item = content_list[i]
                num_contents = len(content_item["type"])
                for j in range(num_contents):
                    content = {
                        "index": content_item["index"][j],
                        "text": content_item["text"][j],
                        "type": content_item["type"][j]
                    }
                    message["content"].append(content)
                messages.append(message)
            example["messages"] = messages
            # Ensure images are in a list
            if not isinstance(example["images"], list):
                example["images"] = [example["images"]]

            # Resize images
            image_size = (1120, 1120)
            resized_images = []
            for image in example["images"]:
                if image.size != image_size:
                    image = image.resize(image_size, resample=Image.BICUBIC)
                resized_images.append(image)
            example["images"] = resized_images

            if idx < 3:  # Only print the first 3 examples
                print(f"Example {idx} after modification:")
                print("messages:", example["messages"])
                print("images:", example["images"])

        # Get the texts and images, and apply the chat template
        texts = [processor.apply_chat_template(example["messages"], tokenize=False) for example in examples]
        images = [example["images"] for example in examples]
        if isinstance(model, LlavaForConditionalGeneration):
            # LLava1.5 does not support multiple images
            images = [image[0] for image in images]

        # Tokenize the texts and process the images
        batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

        # The labels are the input_ids, and we mask the padding tokens in the loss computation
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100  #
        # Ignore the image token index in the loss computation (model specific)
        image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
        labels[labels == image_token_id] = -100
        batch["labels"] = labels

        return batch

    ################
    # Dataset
    ################
    dataset = load_dataset(script_args.dataset_name)

    ################
    # Training
    ################
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split],
        tokenizer=processor.tokenizer,
        peft_config=get_peft_config(model_config),
    )

    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
        if trainer.accelerator.is_main_process:
            processor.push_to_hub(training_args.hub_model_id)
