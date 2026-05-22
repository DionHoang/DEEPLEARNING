import torch
from src.config import NUM_LABELS
from src.models import PhoBERTNER, PhoBERTLoRANER

def test_models():
    print("=========================================")
    print("Starting Model Architecture Verification")
    print("=========================================")
    
    # 1. Test Standard PhoBERTNER
    print("\n[1/3] Initializing PhoBERTNER...")
    try:
        model = PhoBERTNER(num_labels=NUM_LABELS)
        print("✓ PhoBERTNER initialized successfully.")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Trainable parameters: {trainable_params:,}")
        
        # Mock inputs
        batch_size = 2
        seq_length = 32
        mock_input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        mock_attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)
        mock_labels = torch.randint(0, NUM_LABELS, (batch_size, seq_length))
        
        # Test Forward Pass
        print("  - Running mock forward pass...")
        outputs = model(mock_input_ids, mock_attention_mask, mock_labels)
        print(f"✓ Forward pass successful! Output logits shape: {outputs['logits'].shape}")
        print(f"✓ Loss calculated: {outputs['loss'].item():.4f}")
        
    except Exception as e:
        print(f"✗ PhoBERTNER failed: {e}")

    # 2. Test PhoBERTLoRANER
    print("\n[2/3] Initializing PhoBERTLoRANER...")
    try:
        model_lora = PhoBERTLoRANER(num_labels=NUM_LABELS)
        print("✓ PhoBERTLoRANER initialized successfully.")
        
        # Check trainable parameters statistics
        model_lora.print_trainable_parameters()
        
        # Mock inputs
        batch_size = 2
        seq_length = 32
        mock_input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        mock_attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)
        mock_labels = torch.randint(0, NUM_LABELS, (batch_size, seq_length))
        
        # Test Forward Pass
        print("  - Running mock forward pass...")
        outputs_lora = model_lora(mock_input_ids, mock_attention_mask, mock_labels)
        print(f"✓ LoRA Forward pass successful! Output logits shape: {outputs_lora['logits'].shape}")
        print(f"✓ Loss calculated: {outputs_lora['loss'].item():.4f}")
        
    except Exception as e:
        print(f"✗ PhoBERTLoRANER failed: {e}")
        
    print("\n=========================================")
    print("Verification Completed")
    print("=========================================")

if __name__ == "__main__":
    test_models()
