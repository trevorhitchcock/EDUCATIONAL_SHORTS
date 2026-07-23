from educational_shorts.client import ask_llm
from educational_shorts.schemas import PlanetList

SYSTEM_PROMPT = """
Return only valid JSON.
"""

def run_tests():

    result = ask_llm(
        system_prompt=SYSTEM_PROMPT,
        user_prompt="List the first three planets.",
        schema=PlanetList,
    )

    print(result)


if __name__ == "__main__":
    run_tests()