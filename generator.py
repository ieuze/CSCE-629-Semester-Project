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
    parser.add_argument('--element-min', type=int, default=-5, help='Minimum value for elements in P matrix. Default: -100')
    parser.add_argument('--element-max', type=int, default=5, help='Maximum value for elements in P matrix. Default: 100')

    # Arguments specific to annealing_generator.py or adaptive_annealing.py
    parser.add_argument('--t-max', type=float, default=100, help='Initial annealing temperature.')
    parser.add_argument('--t-min', type=float, help='Final annealing temperature.')
    parser.add_argument('--alpha', type=float, default=0.95, help='Cooling rate (multiplier).')
    parser.add_argument('--iter-per-temp', type=int, default=100, help='Iterations per temperature level.')
    parser.add_argument('--workers', type=int, default=12, help='Number of parallel annealing runs.')
    
    # Arguments specific to adaptive model (currently unused, but keep for potential future use)
    # parser.add_argument('--strategy', type=str, choices=['heuristic', 'sampled', 'adaptive', 'auto'],
    #                   default='auto', help='Surrogate strategy for m > 3. Default: auto')
    # parser.add_argument('--sample-rate', type=float, default=0.1,
    #                   help='Sample rate for surrogate model (m > 3). Default: 0.1')
    # parser.add_argument('--verify-final', action='store_true',
    #                   help='Verify final result with exact method for m > 3')

    # New arguments for choosing method when m > 3 and for random search
    parser.add_argument('--large-m-method', type=str, choices=['annealing', 'random'], default='annealing',
                        help='Method to use when m > 3. Default: annealing')
    parser.add_argument('--num-samples', type=int, default=10000,
                        help='Number of random samples to check (used only if --large-m-method=random). Default: 10000')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility (used only if --large-m-method=random).')

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
        if args.large_m_method == 'annealing':
            target_script = 'annealing_generator.py'
            print(f"--- m = {args.m} > 3, method='annealing', calling {target_script} ---")
            base_command.append(target_script)

            # Add core arguments
            base_command.extend(['-k', str(args.k)])
            base_command.extend(['-n', str(args.n)])
            base_command.extend(['-m', str(args.m)])

            # Add optional arguments relevant to annealing
            if args.element_min is not None:
                base_command.extend(['--element-min', str(args.element_min)])
            if args.element_max is not None:
                base_command.extend(['--element-max', str(args.element_max)])
            if args.t_max is not None:
                base_command.extend(['--t-max', str(args.t_max)])
            if args.t_min is not None:
                base_command.extend(['--t-min', str(args.t_min)])
            if args.alpha is not None:
                base_command.extend(['--alpha', str(args.alpha)])
            if args.iter_per_temp is not None:
                base_command.extend(['--iter-per-temp', str(args.iter_per_temp)])
            if args.workers is not None:
                # Pass the workers argument to annealing script
                base_command.extend(['--workers', str(args.workers)])

            # Add any unknown arguments (might be relevant for annealing)
            base_command.extend(unknown_args)

        elif args.large_m_method == 'random':
            target_script = 'random_generator.py'
            print(f"--- m = {args.m} > 3, method='random', calling {target_script} ---")
            base_command.append(target_script)

            # Add core arguments
            base_command.extend(['-k', str(args.k)])
            base_command.extend(['-n', str(args.n)])
            base_command.extend(['-m', str(args.m)])

            # Add optional arguments relevant to random search
            if args.element_min is not None:
                base_command.extend(['--element-min', str(args.element_min)])
            if args.element_max is not None:
                base_command.extend(['--element-max', str(args.element_max)])
            if args.num_samples is not None:
                base_command.extend(['--num-samples', str(args.num_samples)])
            if args.seed is not None:
                base_command.extend(['--seed', str(args.seed)])
            # Note: progress_interval from random_generator is not exposed here, uses its default.
            # Note: workers arg from annealing is not used by random_generator (currently single-threaded)

            # Add any unknown arguments (less likely to be relevant for random, but pass just in case)
            base_command.extend(unknown_args)

        else:
             # This case should not be reachable due to 'choices' in argument definition
             parser.error(f"Unknown large-m-method: {args.large_m_method}")
             sys.exit(1) # Explicit exit after error

        # Check if the selected target script exists
        if not os.path.exists(target_script):
            print(f"Error: {target_script} not found. Please ensure the script exists.")
            sys.exit(1)

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
