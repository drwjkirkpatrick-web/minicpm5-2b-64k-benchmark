def fibonacci(n, memo=None):
    """
    Returns the nth Fibonacci number using memoization.
    
    Args:
        n (int): The position in the Fibonacci sequence (0-indexed).
        memo (dict, optional): Internal cache for memoization.
        
    Returns:
        int: The nth Fibonacci number.
        
    Raises:
        ValueError: If n is negative.
    """
    if n < 0:
        raise ValueError("n must be a non-negative integer")
    
    if memo is None:
        memo = {}
    
    if n in memo:
        return memo[n]
    
    if n == 0:
        result = 0
    elif n == 1:
        result = 1
    else:
        result = fibonacci(n - 1, memo) + fibonacci(n - 2, memo)
    
    memo[n] = result
    return result
