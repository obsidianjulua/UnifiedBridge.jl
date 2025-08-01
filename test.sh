#!/usr/bin/bash julia

import Pkg

include("UnifiedBridge.jl")

using Revise
using Distributed
using .UnifiedBridge


# 1. run_simple_bash
println("Test 1a: ", run_simple_bash("echo hello world"))

# 2. run_with_args
println("Test 1b: ", run_with_args("echo", ["this", "works"]))

# 3. spawn_worker
fut = spawn_worker("echo spawned")
println("Test 1c: ", fetch(fut))

# 4. capture_output
out = capture_output("echo captured")
println("Test 1d: ", out)

# 5. execute_full
stdout, stderr, code = execute_full("ls /tmp")
println("Test 1e: ", (stdout, stderr, code))

# Set 2: Bash→Julia Arg Conversion

# 6. bash_to_julia_args
args1 = ["-X", "POST", "https://api.com"]
dict1 = bash_to_julia_args(args1)
println("Test 2a: ", dict1)

# 7. julia_to_bash_args
recon = julia_to_bash_args(dict1)
println("Test 2b: ", recon)

# 8. Validate round-trip consistency
println("Test 2c: ", args1 == recon)

# 9. Option edge-case
dict2 = bash_to_julia_args(["--json", "-a", "val", "file.txt"])
println("Test 2d: ", dict2)

# 10. Convert back
println("Test 2e: ", julia_to_bash_args(dict2))

# Set 3: Learning System Integration

# 11. Initial stats
println("Test 3a: ", get_learning_stats())

# 12. Learn pattern
n = learn_args_pattern!("grep", ["-i", "pattern", "file.txt"])
println("Test 3b: Learned count: ", n)

# 13. Recheck stats
println("Test 3c: ", get_learning_stats())

# 14. Predict arg count
println("Test 3d: ", predict_args_count("grep"))

# 15. Expand args
println("Test 3e: ", expand_args_from_pattern("grep", ["-i"]))

# Set 4: Dispatch Verification

# 16. Julia command route
println("Test 4a: ", execute_auto_command("run_simple_bash", Dict("positional"=>["echo hello"])))

# 17. Bash command route
println("Test 4b: ", execute_auto_command("echo", Dict("positional"=>["hey there"])))

# 18. Fallback bash
println("Test 4c: ", execute_auto_command("whoami", Dict("positional"=>[])))

# 19. Unknown fallback
println("Test 4d: ", execute_auto_command("idontexist", Dict()))

# 20. Julia wrapper for capture_output
println("Test 4e: ", execute_auto_command("capture_output", Dict("positional"=>["echo test"])))

# Set 5: Full Learning Execution

# 21. Clean test with learning
result, learned = execute_with_learning("echo", ["Learn", "me"])
println("Test 5a: Result: ", result)
println("Test 5b: Learned Args: ", learned)

# 22. Check signature update
println("Test 5c: Symbol signature: ", get_symbol("echo").signature)

# 23. Try again with added arg
result2, learned2 = execute_with_learning("echo", ["Learn", "me", "again"])
println("Test 5d: ", result2)

# 24. Recheck updated learning state
println("Test 5e: ", get_learning_stats())
