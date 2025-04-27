import torch
import numpy as np
import os
import argparse
from common.models import load_model
from common.utils import get_device

def compare_models(original_path, traced_path, device, test_points=None, tolerance=1e-5):
    """
    Compare the outputs of the original model and the JIT-traced model for consistency
    
    Args:
        original_path: Path to the original PyTorch model (.pt)
        traced_path: Path to the traced model saved with torch.jit.trace (.pt)
        device: Computing device ('cuda' or 'cpu')
        test_points: Test point coordinates, defaults to randomly generated points
        tolerance: Output difference tolerance
        
    Returns:
        True if successful, False if failed
    """
    print(f"Loading original model: {original_path}")
    try:
        # Load original model
        original_model = load_model(original_path, device)
        original_model.eval()
    except Exception as e:
        print(f"Failed to load original model: {e}")
        return False
    
    print(f"Loading JIT model: {traced_path}")
    try:
        # Load JIT model
        jit_model = torch.jit.load(traced_path, map_location=device)
        jit_model.eval()
    except Exception as e:
        print(f"Failed to load JIT model: {e}")
        return False
    
    # Determine input dimension
    input_dim = 3  # Default to 3D
    if hasattr(original_model, 'meta') and len(original_model.meta) > 0:
        input_dim = original_model.meta[0]
        print(f"Detected input dimension from model metadata: {input_dim}")
    
    # Create test inputs - individual points
    if test_points is None:
        # Generate random test points in unit cube
        n_test_points = 10
        test_points = torch.rand(n_test_points, input_dim).to(device) * 2 - 1  # Range [-1, 1]
    else:
        test_points = test_points.to(device)
    
    # Test forward pass for individual points
    print(f"\nTesting forward pass for {test_points.shape[0]} individual points...")
    with torch.no_grad():
        original_outputs = []
        jit_outputs = []
        is_consistent = True
        
        for i, point in enumerate(test_points):
            # Original model forward pass
            original_output = original_model(point).cpu().numpy()
            
            # JIT model forward pass
            jit_output = jit_model(point).cpu().numpy()
            
            original_outputs.append(original_output)
            jit_outputs.append(jit_output)
            
            # Output details
            print(f"Test point {i+1}: Coordinates = {point.cpu().numpy()}")
            print(f"  - Original model output: {original_output.flatten()}")
            print(f"  - JIT model output:      {jit_output.flatten()}")
            
            # Compare outputs
            diff = np.abs(original_output - jit_output).max()
            print(f"  - Maximum absolute error: {diff}")
            
            if diff > tolerance:
                print(f"  [WARNING] Output difference exceeds allowed tolerance ({tolerance})!")
                is_consistent = False
            else:
                print(f"  [PASS] Outputs match (error within tolerance {tolerance})")
    
    # Test batch processing
    print(f"\nTesting batch mode ({test_points.shape[0]} points)...")
    batch_points = test_points
    
    with torch.no_grad():
        try:
            # Original model batch forward pass
            original_batch_output = original_model(batch_points).cpu().numpy()
            
            # JIT model batch forward pass
            jit_batch_output = jit_model(batch_points).cpu().numpy()
            
            # Compare batch outputs
            batch_diff = np.abs(original_batch_output - jit_batch_output).max()
            print(f"Batch maximum absolute error: {batch_diff}")
            
            if batch_diff > tolerance:
                print(f"[WARNING] Batch output difference exceeds allowed tolerance ({tolerance})!")
                is_consistent = False
            else:
                print(f"[PASS] Batch outputs match (error within tolerance {tolerance})")
                
        except Exception as e:
            print(f"Batch testing failed: {e}")
            print("[WARNING] Failing batch test may indicate the model won't handle batches correctly in C++.")
    
    # Summary
    print("\nOverall Assessment:")
    all_diffs = [np.abs(o - j).max() for o, j in zip(original_outputs, jit_outputs)]
    max_diff = max(all_diffs)
    avg_diff = sum(all_diffs) / len(all_diffs)
    
    print(f"Maximum output difference: {max_diff}")
    print(f"Average output difference: {avg_diff}")
    
    if max_diff <= tolerance:
        print("[PASS] Model outputs are consistent and safe for C++ deployment.")
        is_consistent = True
    else:
        print("[FAIL] Model outputs show significant differences, may cause issues in C++ deployment.")
        is_consistent = False
    
    return is_consistent

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Model Consistency Test",
        description="Compare outputs between original PyTorch model and JIT-traced model"
    )
    
    parser.add_argument("original_model", type=str, help="Original model path (.pt file)")
    parser.add_argument("--jit_model", type=str, default=None, 
                        help="JIT model path (.pt file, defaults to cpp/model_final.pt if not provided)")
    parser.add_argument("--tolerance", type=float, default=1e-5, help="Output difference tolerance")
    parser.add_argument("--cpu", action="store_true", help="Force CPU for testing")
    parser.add_argument("--num_points", type=int, default=10, help="Number of test points")
    
    args = parser.parse_args()
    
    device = "cpu" if args.cpu else get_device(False)
    print(f"Using device: {device}")
    
    # Use default path if JIT model path not provided
    jit_model_path = args.jit_model
    if jit_model_path is None:
        original_dir = os.path.dirname(args.original_model)
        jit_model_path = os.path.join(original_dir, "cpp", "model_final.pt")
        if not os.path.exists(jit_model_path):
            print(f"Default JIT model path not found: {jit_model_path}")
            print("Please specify JIT model path with --jit_model")
            exit(1)
    
    # Generate test points
    test_points = torch.rand(args.num_points, 3).to(device) * 2 - 1  # Default 3D, range [-1,1]
    
    # Run comparison
    success = compare_models(args.original_model, jit_model_path, device, 
                            test_points=test_points, tolerance=args.tolerance)
    
    if success:
        exit(0)
    else:
        exit(1)