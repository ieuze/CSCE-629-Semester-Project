import argparse
import subprocess
import sys
import os

def main():
    parser = argparse.ArgumentParser(
        description="Dispatcher script to generate a systematic generator matrix G=[I|P]. "
                    "Calls brute force for m <= 3 and adaptive annealing for m > 3.",
        # Allow unknown arguments to pass through to sub-scripts
        # argument_default=argparse.SUPPRESS # This might hide args from help, use parse_known_args instead
    )

    # Arguments common to both or potentially needed by annealing
    parser.add_argument('-k', type=int, default=3, help='Number of message bits (rows).')
    parser.add_argument('-n', type=int, default=6, help='Number of codeword bits (columns).')
    parser.add_argument('-m', type=int, default=2, help='m-height parameter.')

    # Arguments primarily for annealing, but might be used by brute force too
    parser.add_argument('--element-min', type=int, default=-1, help='Minimum value for elements in P matrix. Default: -1')
    parser.add_argument('--element-max', type=int, default=1, help='Maximum value for elements in P matrix. Default: 1')

    # Arguments specific to annealing_generator.py or adaptive_annealing.py
    parser.add_argument('--t-max', type=float, help='Initial annealing temperature.')
    parser.add_argument('--t-min', type=float, help='Final annealing temperature.')
    parser.add_argument('--alpha', type=float, help='Cooling rate (multiplier).')
    parser.add_argument('--iter-per-temp', type=int, help='Iterations per temperature level.')
    parser.add_argument('--workers', type=int, help='Number of parallel annealing runs.')
    
    # Arguments specific to adaptive model
    parser.add_argument('--strategy', type=str, choices=['heuristic', 'sampled', 'adaptive', 'auto'], 
                      default='auto', help='Surrogate strategy for m > 3. Default: auto')
    parser.add_argument('--sample-rate', type=float, default=0.1, 
                      help='Sample rate for surrogate model (m > 3). Default: 0.1')
    parser.add_argument('--verify-final', action='store_true', 
                      help='Verify final result with exact method for m > 3')

    # Parse known args first to decide the script, then pass remaining if necessary
    args, unknown_args = parser.parse_known_args()

    # Basic validation needed for dispatch logic
    if args.k is None or args.n is None or args.m is None:
        parser.error("Arguments -k, -n, and -m are required.")
    if args.n <= args.k:
        parser.error(f"n ({args.n}) must be greater than k ({args.k}).")
    if not (1 <= args.m <= args.n - 1):
         parser.error(f"m ({args.m}) must be between 1 and n-1 ({args.n-1}).")


    base_command = [sys.executable] # Use the same python interpreter

    if args.m <= 3:
        target_script = 'brute_force_generator.py'
        print(f"--- m = {args.m} <= 3, attempting to call {target_script} for exact search ---")
        base_command.append(target_script)

        # Add required arguments for brute force (assuming similar core args)
        base_command.extend(['-k', str(args.k)])
        base_command.extend(['-n', str(args.n)])
        base_command.extend(['-m', str(args.m)])

        # Add optional arguments if provided
        if args.element_min is not None:
            base_command.extend(['--element-min', str(args.element_min)])
        if args.element_max is not None:
            base_command.extend(['--element-max', str(args.element_max)])

        # Add any unknown arguments that might be specific to brute force
        base_command.extend(unknown_args)

        if not os.path.exists(target_script):
            print(f"Error: {target_script} not found. Please create this script.")
            sys.exit(1)

    else: # m > 3
        target_script = 'hybrid_annealing.py'
        print(f"--- m = {args.m} > 3, calling {target_script} for hybrid annealing ---")
        base_command.append(target_script)

        # Pass only core arguments relevant to hybrid annealing
        base_command.extend(['-k', str(args.k)])
        base_command.extend(['-n', str(args.n)])
        base_command.extend(['-m', str(args.m)])

        # Pass element range if provided
        if args.element_min is not None:
            base_command.extend(['--element-min', str(args.element_min)])
        if args.element_max is not None:
            base_command.extend(['--element-max', str(args.element_max)])

        # Add any unknown arguments (allows passing hybrid-specific args like --surrogate-t-max)
        base_command.extend(unknown_args)

    print(f"Executing command: {' '.join(base_command)}")
    try:
        # Execute the command
        # Check=True raises CalledProcessError if the script exits with non-zero status
        # Capture_output=False lets the script print directly to console
        process = subprocess.run(base_command, check=True, text=True)
        print(f"--- {target_script} finished successfully ---")

    except FileNotFoundError:
        print(f"Error: The script '{target_script}' was not found.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Error: {target_script} exited with status {e.returncode}.")
        # stderr might be None if capture_output=False, but error message is usually printed by the script itself
        sys.exit(e.returncode)
    except Exception as e:
        print(f"An unexpected error occurred while trying to run {target_script}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
