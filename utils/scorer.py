"""Scorer"""
import nltk
import nltk
from nltk.translate.bleu_score import SmoothingFunction
from rouge_score import rouge_scorer

smoothie = SmoothingFunction().method4

class BleuScorer(object):
    """Blue scorer class"""
    def __init__(self):
        self.results = []
        self.results_meteor = []
        self.score = 0
        self.meteor_score = 0
        self.instances = 0
        self.meteor_instances = 0

    def example_score(self, reference, hypothesis):
        """Calculate blue score for one example"""
        """Calculate blue score for one example"""
        bleu_1 = nltk.translate.bleu_score.sentence_bleu(reference, hypothesis,weights=(1,0,0,0),smoothing_function=SmoothingFunction().method4)
        bleu_4 = nltk.translate.bleu_score.sentence_bleu(reference, hypothesis,weights=(0.25,0.25,0.25,0.25),smoothing_function=SmoothingFunction().method4)
        return bleu_1, bleu_4
    
    def example_score_rouge(self, reference, hypothesis):
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
        scores = []
        for i in reference:
            scores.append(scorer.score(i,hypothesis)['rougeL'][-1])
        return np.max(scores) #best
    
    def example_score_meteor(self, reference, hypothesis):
        """Calculate blue score for one example"""
        return nltk.translate.meteor_score.meteor_score(reference,hypothesis)

    def data_score(self, data, predictor):
        """Score complete list of data"""
        results_prelim = []
        for example in tqdm_notebook(data):

            src = example[0]
            reference = [[string.lower() for string in sublist] for sublist in example[1]]
            
            hypothesis = predictor.predict(src)
            bleu_1,bleu_4 = self.example_score(reference, hypothesis)
            meteor_score = self.example_score_meteor([' '.join(i) for i in reference], ' '.join(hypothesis))
            rouge_score = self.example_score_rouge([' '.join(i) for i in reference], ' '.join(hypothesis))
            
            f = open("result.txt", "a")
            f.write('Question: '+" ".join(src)+'\n')
            for i in range(len(reference)):
                f.write('Reference_{}: '.format(i)+" ".join(reference[i])+'\n')
            f.write('Hypothesis: '+" ".join(hypothesis)+'\n')
            f.write('BLEU-1: '+ str(bleu_1*100)+'\n')
            f.write('BLEU-4: '+str(bleu_4*100)+'\n')
            f.write('METEOR: '+str(meteor_score*100)+'\n')
            f.write('ROUGE-L: '+str(rouge_score*100)+'\n\n')
            f.close()
            
            results_prelim.append({
                'question': '"' + str(src) + '"',
                'reference': reference,
                'hypothesis': hypothesis,
                'bleu_1': bleu_1,
                'bleu_4': bleu_4,
                'meteor_score': meteor_score,
                'rouge_score': rouge_score,
                
            })
    
        results = [max((v for v in results_prelim if v['question'] == x), key=lambda y:y['bleu_1']) for x in set(v['question'] for v in results_prelim)] 

        with open('result_output.txt', 'w') as f:
            for elem in results:
                f.write("%s\n" % elem)
                self.results.append(elem)
                self.score += elem['bleu_1']
                self.bleu_4 += elem['bleu_4']
                self.meteor_score += elem['meteor_score']
                self.rouge_score += elem['rouge_score']
                self.instances += 1
        return self.score / self.instances, self.bleu_4 / self.instances, self.meteor_score / self.instances, self.rouge_score / self.instances


    def average_score(self):
        """Return bleu average score"""
        return self.score / self.instances, self.bleu_4 / self.instances
    
    def average_rouge_score(self):
        """Return bleu average score"""
        return self.rouge_score / self.instances

     def average_meteor_score(self):
        """Return meteor average score"""
        return self.meteor_score/self.instances
    
    def reset(self):
        """Reset object properties"""
        self.results = []
        self.results_meteor = []
        self.score = 0
        self.meteor_score = 0
        self.instances = 0
        self.meteor_instances = 0
