import re
import numpy as np
import torch

class Jamo():
    
    chosung = ['ㄱ','ㄲ','ㄴ','ㄷ','ㄸ','ㄹ','ㅁ','ㅂ','ㅃ','ㅅ',
               'ㅆ','ㅇ','ㅈ','ㅉ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ']
    jungsung = ['ㅏ','ㅐ','ㅑ','ㅒ','ㅓ','ㅔ','ㅕ','ㅖ','ㅗ','ㅘ','ㅙ',
                'ㅚ','ㅛ','ㅜ','ㅝ','ㅞ','ㅟ','ㅠ','ㅡ','ㅢ','ㅣ']
    jongsung = ['','ㄱ','ㄲ','ㄳ','ㄴ','ㄵ','ㄶ','ㄷ','ㄹ','ㄺ','ㄻ','ㄼ','ㄽ','ㄾ',
                'ㄿ','ㅀ','ㅁ','ㅂ','ㅄ','ㅅ','ㅆ','ㅇ','ㅈ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ']    
    
    def __init__(self):    
        self.chosung_dic = {key:value for key,value in enumerate(Jamo.chosung)}
        self.jungsung_dic = {key:value for key,value in enumerate(Jamo.jungsung)} 
        self.jongsung_dic = {key:value for key,value in enumerate(Jamo.jongsung)}
    
    def split_jamo(self, sentence, remove_jongsung=True, remove_blank=True):
        jamo_lst = []
        for syllable in list(self.preprocess(sentence, remove_blank)):
            if re.match(r'[가-힣]',syllable):
                syllable_code = ord(syllable)
                chosung_idx = int((syllable_code-44032)/588)
                jungsung_idx = int((syllable_code-44032-(chosung_idx*588))/28)
                jongsung_idx = int((syllable_code-44032-(chosung_idx*588)-(jungsung_idx*28)))
        
                jamo_lst.append(self.chosung_dic[chosung_idx])
                jamo_lst.append(self.jungsung_dic[jungsung_idx])
                jamo_lst.append(self.jongsung_dic[jongsung_idx])
            else:
                jamo_lst.append(syllable.lower())
        
        if remove_jongsung:
            jamo_lst = [jamo for jamo in jamo_lst if len(jamo) > 0]
        
        return jamo_lst

    @staticmethod
    def preprocess(text, remove_blank=True):
        if remove_blank: 
            preprocessed_stc = text.replace(' ','')
        else:
            eojeol_lst = [eojeol.strip() for eojeol in text.split(' ') if not len(eojeol)==0]
            preprocessed_stc = '▁'.join(eojeol_lst)
        return preprocessed_stc


def get_score(logits, labels):
    logits_flat = np.argmax(logits, axis=1).flatten()
    labels_flat = labels.flatten() # not essential
    
    TP = np.sum((logits_flat == 1) & (labels_flat == 1))
    TN = np.sum((logits_flat == 0) & (labels_flat == 0))
    FP = np.sum((logits_flat == 1) & (labels_flat == 0))
    FN = np.sum((logits_flat == 0) & (labels_flat == 1))

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    
    return {
        'accuracy':accuracy,
        'precision':precision,
        'recall':recall, 
    }


def test(model, test_dataloader, args):
    
    total_accuracy = 0
    total_precision = 0
    total_recall = 0
    
    model.eval()    
    for _, batch in enumerate(test_dataloader):
            
        input_ids = batch['text'].to(args.device)
        label = batch['label'].to(args.device)
            
        with torch.no_grad():
            logits = model(input_ids)
        
        logits = logits.detach().cpu().numpy()
        label = label.to('cpu').numpy()
        scores = get_score(logits, label)
                
        total_accuracy += scores['accuracy']
        total_precision += scores['precision']
        total_recall += scores['recall']
    
    avg_accuracy = total_accuracy / len(test_dataloader)
    avg_precision = total_precision / len(test_dataloader)
    avg_recall = total_recall / len(test_dataloader)
    
    return {
        'accuracy':avg_accuracy,
        'precision':avg_precision,
        'recall':avg_recall,
    }