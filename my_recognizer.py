import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    # TODO implement the recognizer
     # TODO implement the recognizer
    Xlength_dict ={}
    
    for word_idx in range(len(test_set.get_all_Xlengths())):
        best_prob = float('-inf')
        best_word = None      
        X,length = test_set.get_item_Xlengths(word_idx)
        
        for word,model in models.items():
            try:
                logL = model.score(X,length)
                Xlength_dict[word] = logL
            except:
                logL = float('-inf')
                Xlength_dict[word] = logL
            
            if Xlength_dict[word] > best_prob:
                best_prob = Xlength_dict[word]
                best_word = word
                
        guesses.append(best_word)                
        probabilities.append(Xlength_dict)
    
    #print('probabilities: {}'.format(probabilities))    
    #print('guesses: {}'.format(guesses))
    
    return probabilities,guesses
