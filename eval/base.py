import re
from tasks.utils import evaluate, evaluate_fol_manually


def postprocess_generation(generation):
    matches = re.findall(r"<EVALUATE>(.*?)</EVALUATE>", generation, re.DOTALL)
    block = matches[-1].strip() if matches else generation.strip()
    flag = "FOL:"
    fol_lines = [
        line.replace(flag, "").strip()
        for line in block.split("\n")
        if flag in line
    ]
    premises, conclusion = fol_lines[:-1], fol_lines[-1]

    try:
        resp = evaluate_fol_manually(premises, conclusion)
        # print("Evaluation result:", resp)
        # if resp not in ("True", "False", "Uncertain"):
        #     print("Unexpected evaluation result:", resp)
        # if resp in "contradiction": 
        #     print(f"Contradiction found in: {premises} and {conclusion}")
        return resp
    except Exception as e:
        # print the exact line that can produce the error of Error evaluating FOL: Unexpected token: 'consistently'.  Expected token ')'.
        print("Error evaluating FOL:", e)
        return "Error"
    
# def postprocess_generation(generation):
#     matches = re.findall(r"<EVALUATE>(.*?)</EVALUATE>", generation, re.DOTALL)
#     block = matches[-1].strip() if matches else generation.strip()
#     flag = "FOL:"
#     fol_lines = [
#         line.replace(flag, "").strip()
#         for line in block.split("\n")
#         if flag in line
#     ]
#     premises, conclusion = fol_lines[:-1], fol_lines[-1]

#     try:
#         resp = evaluate(premises, conclusion)
#         print("Evaluation result:", resp)
#         if resp not in ("True", "False", "Uncertain"):
#             print("Unexpected evaluation result:", resp)
#         return resp
#     except Exception as e:
#         print("Error evaluating FOL:", e)
#         return "Error"
