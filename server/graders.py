import math

def dcg(scores):
    return sum(
        rel / math.log2(i + 2)  # i+2 because index starts at 0
        for i, rel in enumerate(scores)
    )
def ndcg_with_truth(predicted, truth_count):
    dcg_val = dcg(predicted)

    # Ideal list = all relevant first
    ideal = [1]*truth_count + [0]*(len(predicted)-truth_count)
    idcg_val = dcg(ideal)

    return dcg_val / idcg_val if idcg_val > 0 else 0.0


def easy_grader(task_ranking:list, task_length:int):
    task_ranking = list(map(lambda x: 1 if x >= 0.99 else 0, task_ranking))
    return max(0.01, min(0.99, ndcg_with_truth(task_ranking, task_length)))

def medium_grader(task_ranking:list, task_length:int):
    task_ranking = list(map(lambda x: 1 if x >= 0.99 else 0, task_ranking))
    return max(0.01, min(0.99, ndcg_with_truth(task_ranking, task_length)))

def hard_grader(task_ranking:list, task_length:int):
    task_ranking = list(map(lambda x: 1 if x >= 0.99 else 0, task_ranking))
    return max(0.01, min(0.99, ndcg_with_truth(task_ranking, task_length)))


# if __name__ == "__main__":
#     # test graders
#     # print(easy_grader([0.99, 0.60, 0.99, 0.60, 0.99], 3)) # 0.8854598815714874
#     # print(easy_grader([0.99, 0.99, 0.99], 3)) # 0.99
#     # print(easy_grader([0.99, 0.40, 0.99, 0.40, 0.99], 3)) # 0.8854598815714874