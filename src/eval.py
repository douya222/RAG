from bert_score import score
import jieba
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
from rouge_chinese import Rouge

reference = ['藜麦常见虫害有象甲虫、金针虫、蝼蛄、黄条跳甲、横纹菜蝽、萹蓄齿胫叶甲、潜叶蝇、蚜虫、夜蛾等。防治方法：可每亩用3%的辛硫磷颗粒剂2-2.5千克于耕地前均匀撒施，随耕地翻入土中，也可以每亩用40%的辛硫磷乳油250毫升，加水1-2千克，拌细土20-25千克配成毒土，撒施地面翻入土中，防治地下害虫']
llms = ['黎虫害的防治方法如下 :1。选择抗虫品种 :选择抗虫性能较好的品种,能够有效降低虫害的发生率。合理种植密度:合理密植可以减少叶片接触，降低虫害的发生率3。定期检查:定期检查作物生长情况，及时发现虫害，采取措施防治。生物防治:利用天敌、病原菌、昆虫生态调控剂等生物防治措施，降低虫害的发生率。5，化学防治:合理使用农药,控制农药使用量,防止农药过量使用导致虫害抗药性增强。物理防治:利用遮阳网、防虫网等物理措施，减少虫害的发生综上所述，黎虫害的防治需要采取多种措施，根据具体情况选择合适的防治方法。']
rags = ['针对黎怎么防治虫害的问题，根据提供的背景知识，可以采取以下措施:1.叶斑病: 使用12.5%的烯哗醇可湿性粉剂3000-4000倍液喷雾防治，一般防治1-2次即可收到效果。2.金针虫、蛟站、黄条跳甲、横纹菜蜻、篇蓄齿胫叶甲、潜叶蝇、蚜虫、夜蛾等:可每亩用3%的辛硫磷颗粒剂2-2.5千克于耕地前均匀撒施，随耕地翻入土中，或者每亩用49%的辛硫磷乳油250毫升，加水1-2千克，拌细土20-25千克配成毒土，撒施地面翻入土中，以防治地下害虫。综上，可以通过喷洒叶斑病药剂和使用辛硫磷颗粒剂或乳油进行综合防治虫害']


def bertscore(answer, predicted):
    # 需要和本地bert-chinese model在同一目录
    P, R, F1 = score(answer, predicted, model_type="bert-base-chinese", lang="zh", verbose=True)
    print(f"precision score: {P.mean():.3f}")
    print(f"recall score: {R.mean():.3f}")
    print(f"F1 score: {F1.mean():.3f}")
    res = {
        "P:": P,
        "R:": R,
        "F1:": F1
    }

def bleu(answer, predicted):

    smooth = SmoothingFunction()
    # Tokenize using Jieba
    reference_tokenized = [[list(jieba.cut(sentence)) for sentence in answer]]
    generated_tokenized = [list(jieba.cut(predicted[0]))]
    print("Reference Tokenized:", reference_tokenized)
    print("Generated Tokenized:", generated_tokenized)
    # Calculate BLEU score with a list of references
    bleu_score = corpus_bleu(reference_tokenized, generated_tokenized, smoothing_function=smooth.method1)
    print("BLEU-4 Score:", bleu_score)

def rouge(answer, predicted):
    
    hypothesis = ' '.join(jieba.cut(predicted)) 
    reference = ' '.join(jieba.cut(answer))
    rouge = Rouge()
    scores = rouge.get_scores(hypothesis, reference)
    print("BLEU-4 Score:",scores)
