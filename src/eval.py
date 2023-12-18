import jieba
import json
import torch
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu,SmoothingFunction
from rouge_chinese import Rouge
from bert_score import score

def bertscore(ground_truth, predicted):
    # bert-base-chinese need to be the same dir
    P, R, F1 = score(ground_truth, predicted, model_type="bert-base-chinese", lang="zh", verbose=True)
    print("Precision: {:.4f}".format(torch.mean(P).item()))
    print("Recall: {:.4f}".format(torch.mean(R).item()))
    print("F1 Score: {:.4f}".format(torch.mean(F1).item()))
    return {"precision": torch.mean(P).item(), "recall": torch.mean(R).item(), "f1_score": torch.mean(F1).item()}

def calculate_bleu_scores(ground_truth_list, predicted_list):
    bleu_scores_1 = []
    bleu_scores_2 = []
    smooth = SmoothingFunction()

    for ground_truth, predicted in zip(ground_truth_list, predicted_list):
        reference_tokenized = list(jieba.cut(ground_truth))
        generated_tokenized = list(jieba.cut(predicted))
        bleu_score1 = sentence_bleu([reference_tokenized], generated_tokenized, smoothing_function=smooth.method1)
        # bleu_score2 = corpus_bleu([[reference_tokenized]], [generated_tokenized], smoothing_function=smooth.method1)
        bleu_scores_1.append(bleu_score1)
        # bleu_scores_2.append(bleu_score2)

    # print(sum(bleu_scores_1) / len(bleu_scores_1) * 100)
    # print(sum(bleu_scores_2) / len(bleu_scores_2) * 100)
    average_bleu_score = sum(bleu_scores_1) / len(bleu_scores_1)
    return average_bleu_score * 100

def calculate_rouge_scores(ground_truth_list, predicted_list):
    rouge_scores = []

    for ground_truth, predicted in zip(ground_truth_list, predicted_list):
        hypothesis = ' '.join(jieba.cut(predicted)) 
        reference = ' '.join(jieba.cut(ground_truth))
        rouge = Rouge()
        scores = rouge.get_scores(hypothesis, reference)
        rouge_scores.append(scores)  # Appending the first score in case there are multiple references

    return rouge_scores

def calculate_average_rouge_scores(rouge_scores):
    rouge_1_scores = []
    rouge_2_scores = []
    rouge_l_scores = []

    for scores in rouge_scores:
        rouge_1_scores.append(scores[0]['rouge-1']['f'])
        rouge_2_scores.append(scores[0]['rouge-2']['f'])
        rouge_l_scores.append(scores[0]['rouge-l']['f'])

    # Calculate the average scores for each metric
    avg_rouge_1 = sum(rouge_1_scores) / len(rouge_1_scores)
    avg_rouge_2 = sum(rouge_2_scores) / len(rouge_2_scores)
    avg_rouge_l = sum(rouge_l_scores) / len(rouge_l_scores)

    return {"rouge_1": avg_rouge_1*100, "rouge_2": avg_rouge_2*100, "rouge_l": avg_rouge_l*100}

def main():
    data_path = '/data/user1801004151/research/RAG/src/qa_data/qa_p.json'
    output_path = '/data/user1801004151/research/RAG/src/qa_data/result_glm3.json'

    with open(data_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    ground_truth_list = [item["ground_truth"] for item in data]
    predicted_llm_list = [item["predicted_llm"] for item in data]
    predicted_rag_list = [item["predicted_rag"] for item in data]

    results = {}

    print("==================LLM only===================")
    rouge_scores = calculate_rouge_scores(ground_truth_list, predicted_llm_list)
    results["LLM"] = {
        "bert_scores": bertscore(ground_truth_list, predicted_llm_list),
        "bleu_score": calculate_bleu_scores(ground_truth_list, predicted_llm_list),
        "rouge_scores": calculate_average_rouge_scores(rouge_scores)
    }

    print("==================retrieval + LLM===================")
    rouge_scores = calculate_rouge_scores(ground_truth_list, predicted_rag_list)
    results["retrieval + LLM"] = {
        "bert_scores": bertscore(ground_truth_list, predicted_rag_list),
        "bleu_score": calculate_bleu_scores(ground_truth_list, predicted_rag_list),
        "rouge_scores": calculate_average_rouge_scores(rouge_scores)
    }

    # Write results to a JSON file
    with open(output_path, 'w', encoding='utf-8') as output_file:
        json.dump(results, output_file, ensure_ascii=False, indent=2)

main()
