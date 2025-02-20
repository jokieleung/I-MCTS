import asyncio

from metagpt.roles.di.data_interpreter import DataInterpreter
from metagpt.ext.sela.utils import save_notebook
import os 

async def main(requirement: str = ""):
    di = DataInterpreter()
    await di.run(requirement)
    save_notebook(role=di, save_dir=os.path.join(os.path.dirname(__file__), "notebook_saved"), name="ACV_test_wrong", save_to_depth=False)


if __name__ == "__main__":
    # requirement = "Solve this math problem: The greatest common divisor of positive integers m and n is 6. The least common multiple of m and n is 126. What is the least possible value of m + n?"
    # requirement = """
    # ##User Requirement\nSolve this math problem: The greatest common divisor of positive integers m and n is 6. The least common multiple of m and n is 126. What is the least possible value of m + n?\n\n### Code\n```python\nfrom math import gcd\ndef find_least_m_plus_n(gcd_value, lcm_value):\n    product_mn = gcd_value * lcm_value\n    ab = product_mn // (gcd_value ** 2)\n    min_sum = float('inf')\n    for a in range(1, ab + 1):\n        if ab % a == 0:\n            b = ab // a\n            if gcd(a, b) == 1:\n                m = gcd_value * a\n                n = gcd_value * b\n                m_plus_n = m + n\n                min_sum = min(min_sum, m_plus_n)\n    return min_sum\ngcd_value = 6\nlcm_value = 126\nleast_m_plus_n = find_least_m_plus_n(gcd_value, lcm_value)\nleast_m_plus_n\n\nfrom math import gcd\ndef find_possible_pairs(gcd_value, lcm_value):\n    product_mn = gcd_value * lcm_value\n    ab = product_mn // (gcd_value ** 2)\n    min_sum = float('inf')\n    possible_pairs = []\n    for a in range(1, ab + 1):\n        if ab % a == 0:\n            b = ab // a\n            if gcd(a, b) == 1:\n                m = gcd_value * a\n                n = gcd_value * b\n                m_plus_n = m + n\n                if m_plus_n < min_sum:\n                    min_sum = m_plus_n\n                    possible_pairs = [(m, n)]\n                elif m_plus_n == min_sum:\n                    possible_pairs.append((m, n))\n    return possible_pairs\ngcd_value = 6\nlcm_value = 126\npossible_pairs = find_possible_pairs(gcd_value, lcm_value)\npossible_pairs\n```\n\n### Execution result\n60\n\n[(18, 42), (42, 18)]\n\n
    # ###Instruction
    # Please help me validate the above answer using Python code, and Output in Python Block. Attention: Ensure that the output format is bool.

    # """

    # wrong
    requirement = """
    ##User Requirement\nSolve this math problem: The greatest common divisor of positive integers m and n is 6. The least common multiple of m and n is 126. What is the least possible value of m + n?\n\n### Code\n```python\nfrom math import gcd

def find_min_sum_of_m_n(gcd_m_n, lcm_m_n):
    product_m_n = gcd_m_n * lcm_m_n
    min_sum = float('inf')
    best_pair = (0, 0)
    
    for a in range(1, int(product_m_n**0.5) + 1):
        if product_m_n % a == 0:
            b = product_m_n // a
            if gcd(a, b) == 1:
                m = gcd_m_n * a
                n = gcd_m_n * b
                if m + n < min_sum:
                    min_sum = m + n
                    best_pair = (m, n)
    
    return min_sum, best_pair

# Given values
gcd_m_n = 6
lcm_m_n = 126

# Calculate the minimum sum of m and n
min_sum, best_pair = find_min_sum_of_m_n(gcd_m_n, lcm_m_n)
print(f'The least possible value of m + n is {{min_sum}}, with m = {{best_pair[0]}} and n = {{best_pair[1]}}')
### Execution result\nThe least possible value of m + n is 330, with m = 162 and n = 168\n\n
    ###Instruction
    Please help me validate the above answer using Python code, and Output in Python Block. Attention: Ensure that the output format is bool.

    """
    # answer: 60 (m = 18, n = 42)
    asyncio.run(main(requirement))
    