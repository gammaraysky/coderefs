def table_of_contents():
    """
    CHAPTERS
    Read in text corpus.

    Tokenize into words or sentences.

    Normalize words (lowercase/stemming/lemmatize)

    Part of Speech tagging

    Parsing with Context-free Grammar ### don't really see how to apply this, personally.

    Word Similarity
        Application: Spell-checking using Jaccard Distance or Edit Distance and character ngrams.

    Text Classification/Sentiment Analysis
        CountVectorizer
            make bag of words, i.e convert text into numeric sparse vector matrix.
            (only considers frequency of words, not word order,etc)
        tf-idf (Term Frequency-Inverse Document Frequency)
            allows us to weight terms based on how important they are to a document.
            small tf-idf = very common words, not highly document specific
            high  tf-idf = highly document specific, low commonality across all documents
            smallest coefficients = negative review common words
            highest coefficients  = positive review common words
            (for both cv and tf-idf, can use min_df=k param, which means a word is only made a feature if it appears across at least k documents. this reduces dataset size and training)
        both of these have a problem, word order is disregarded.
        ngrams to capture some sense of word order, but feature list explodes exponentially.

    Semantic Text Similarity
        WordNet organizes information in a hierarchy
        Many similarity measures use the hierarchy in some way
        Verbs, nouns, adjectives all have separate hierarchies
            Path Similarity
            Lin Similarity
            Collocations and Distributional Similarity
            Strength of association between words (Pointwise Mutual Information)
                how frequent do two words occur together?
                log( P(see a and b together) / [P(a) * P(b)])


    Topic Modelling/Text Clustering
        using gensim LDA (Latent Dirichlet Association)
            good for EDA, feature selection for other tasks

    Information Extraction
        Named Entity Recognition
            for well formatted data, can use Regex
            otherwise, use ML
            typically four-class model:
                PER person
                ORG organization
                LOC/GPE location
                Other/Outside any words that do not match the above
        Relation Extraction "Ali created the painting."
        Co-reference Resolution "Anita met Joseph at the market. He surprised her with a rose."
        Question Answering "Who gave Anita the rose?" "What did Ali create?"

    """
    return


def install_nltk():
    # !pip install nltk
    # import nltk
    # nltk.download('popular')
    # nltk.download('gutenberg')
    # nltk.download('genesis')
    # nltk.download('inaugural')
    # nltk.download('nps_chat')
    # nltk.download('webtext')
    # nltk.download('treebank')
    # nltk.download('udhr')
    # nltk.download('wordnet')
    # nltk.download('omw-1.4')
    # nltk.download('punkt')
    # nltk.download('averaged_perceptron_tagger')
    # nltk.download('tagsets')
    # from nltk.book import *
    return

import nltk
texts() # text corporas
sents() # sentences from each text corpora

######!! READ DATA
text= """
China spokeswoman's Taiwan restaurant tweet sparks ridicule online 
Foreign ministry spokeswoman Hua Chunying suggested the popularity of mainland Chinese food in Taiwan proved the island belonged to Beijing. (Photo: AFP/Noel Celis)
BEIJING: A senior Chinese foreign ministry spokeswoman has prompted a storm of ridicule online, after a late-night tweet where she used restaurant listings to assert Beijing's claim over Taiwan.
\"Baidu Maps show that there are 38 Shandong dumpling restaurants and 67 Shanxi noodle restaurants in Taipei,\" spokeswoman Hua Chunying posted on the social media site late on Sunday (Aug 7).
\"Palates don't cheat. #Taiwan has always been a part of China. The long lost child will eventually return home,\" she added.
Hua's tweet comes at the end of a week of tensions around the Taiwan Strait, during which Beijing raged at a trip by United States House Speaker Nancy Pelosi to the island, which China considers part of its territory.
The Chinese government has responded to Pelosi's trip by cancelling a series of talks and cooperation agreements with Washington, as well as deploying fighter jets, warships and ballistic missiles around democratic, self-ruled Taiwan.
Hua's tweet on Sunday appeared to backfire, as thousands of users on Twitter - a site banned in China and only accessible via special VPN software - piled on to poke holes in the top official's logic.
\"There are over 100 ramen restaurants in Taipei, so Taiwan is definitely a part of Japan,\" a Twitter user with the handle \"Marco Chu\" wrote in Hua's replies.
\"Google Maps show that there are 17 McDonalds, 18 KFCs, 19 Burger Kings, and 19 Starbucks in Beijing. Palates don't cheat. #China has always been a part of America. The long lost child will eventually return home,\" Twitter user "@plasticreceiver" wrote in a parody of Hua's tweet.
Others wondered jokingly if Hua's logic meant Beijing could place claims on territories far beyond the Asia Pacific region.
\"There are 29 dumpling houses in the Greater Los Angeles area not to mention 89 noodle restaurants,\" a person tweeting under the name \"Terry Adams\" wrote.
\"Using Hua's logic, LA has always been a part of China.\"
"""



######!! WORD AND SENTENCE TOKENIZATION
def tokenization():
    #! nltk.word_tokenize
    # Sentence splitting is more complicated than just splitting on './!/?'
    # "A gallon of U.S. milk costs #1.50 more in Canada."
    text11 = "Children shouldn't drink a sugary drink before bed."
    print(  len(text11.split(' ')), text11.split(' ')  )
    print(  len(nltk.word_tokenize(text11)), nltk.word_tokenize(text11)  )

    #! nltk.sent_tokenize
    text12 =    "This is the first sentence. \
                A gallon of milk in the U.S. costs $2.99. \
                Is this the third sentence? \
                Yes, it is!"
    sentences = nltk.sent_tokenize(text12)
    print(  len(sentences), sentences  )

    return



######!! NORMALIZATION AS NEEDED
def normalization():
    ###! LOWERCASE
    tokens_lower = nltk.word_tokenize(text.lower()) 

    ###! STEMMING (PORTER STEMMER)
    # Pros and cons to each: some 'stems' are invalid words:
    # run runner running ran runs easily fairly list lists listings listed listicle . Lists be Listed
    # run runner         ran      easili fairli list                       listicl  .       be
    input1 = "run runner running ran runs easily fairly list lists listings listed listicle. Lists be Listed."
    tokens = nltk.word_tokenize(input1)

    porter = nltk.PorterStemmer()
    tokens_stemmed = [porter.stem(t) for t in tokens] 
    dist = FreqDist(tokens_stemmed)

    ###! LEMMATIZATION (WORDNET LEMMATIZER)
    # Pros and cons to each: 
    # Lemmatize is a bit smarter. Capitals treated as names and left as is.
    # run runner running ran runs easily fairly list lists listings listed listicle . Lists be Listed
    # run runner running ran      easily fairly list       listing  listed listicle . Lists be Listed
    WNlemma = nltk.WordNetLemmatizer()
    tokens_lemma = [WNlemma.lemmatize(t) for t in tokens] 
    dist = FreqDist(tokens_lemma)

    ###! DROP STOPWORDS, DROP ~ISALPHA/ISALNUM (drop non alphabetical or numeric)
    from nltk.corpus import stopwords
    # You can view the list of included stop words in NLTK using the code below:
    # there are 40 English words in stopword list.
    import nltk
    from nltk.corpus import stopwords
    stops = set(stopwords.words('english'))
    print(stops)
    # You can do that for different languages, so you can configure for the language you need.
    stops = set(stopwords.words('german'))
    stops = set(stopwords.words('indonesia'))
    stops = set(stopwords.words('portuguese'))
    stops = set(stopwords.words('spanish'))
    data = "All work and no play makes jack dull boy. All work and no play makes jack a dull boy."
    stopWords = set(stopwords.words('english'))
    words = word_tokenize(data)
    wordsFiltered = []

    for w in words:
        if w not in stopWords:
            if w.isalnum():
                wordsFiltered.append(w)

    print(wordsFiltered)

    return

######!! WORD FREQUENCY DISTRIBUTION
def word_freq_dist():
    tokens = nltk.word_tokenize(text) # or use text.split(' ') for simpler content.
    dist = FreqDist(tokens)
    vocab = dist.keys()
    len(set(vocab))

    # list vocab words and counts from most frequent to least
    sorted(dist, key=dist.get, reverse=True) 
    sorted(dist.values(), reverse=True)

    ###? PLOT DISTRIBUTIONS, ETC
    # # let's not consider small words like 'the' or 'a' or fullstops
    # freqwords = [w for w in vocab if len(w) > 5 and dist[w] > 100] 
    # print(  freqwords  )
    # print(  [dist[freqword] for freqword in freqwords]  )

    # import matplotlib.pyplot as plt
    # import numpy as np

    # # PLOT HISTOGRAM OF THE FIRST 50 ITEMS
    # plt.figure(figsize=(20,3))
    # plt.bar( list(dist.keys())[:50], list(dist.values())[:50],  )
    # plt.gca().set_xticklabels(labels=list(dist.keys())[:50], rotation=90);

    # # PLOT HISTOGRAM OF TOP 50 ITEMS SORTED
    # a,b = 0, 50
    # plt.figure(figsize=(20,3))
    # plt.bar( np.sort( list(dist.keys()) , axis=0)[a:b], np.sort( list(dist.values()) , axis=0)[a:b],  )
    # plt.gca().set_xticklabels(labels=np.sort( list(dist.keys()) , axis=0)[a:b], rotation=90);

    return



######!! PART OF SPEECH (POS) TAGGING
def part_of_speech():
    # - CC - Conjunction
    # - CD - Cardinal
    # - DT - Determiner
    # - IN - Preposition
    # - JJ - Adjective
    # - MD - Modal
    # - NN - Noun, common, singular or mass
    #   - NNP - Noun, proper, singular
    #   - NNS - Noun, common, plural
    # - POS - Possessive
    # - PRP - Pronoun
    # - RB - Adverb
    # - SYM - Symbol
    # - VB - Verb 
    #   - VBG: Verb, Present Participle or Gerund
    # - . - Sentence Terminator
    nltk.help.upenn_tagset('NNP')

    text13 = nltk.word_tokenize("Visiting aunts can be a nuisance")
    print(  nltk.pos_tag(text13)  )

    ##! Problem is, English is ambiguous. pos_tag will only parse words as what is default/common in their dataset.
    #? In the above parse, you are the one visiting your aunt.
    # By right, the below parse is legit too:
    # (the aunt is the one visiting)
    # [('Visiting', 'JJ'), ('aunts', 'NNS'), ('can', 'MD'), ('be', 'VB'), ('a', 'DT'), ('nuisance', 'NN')]
    # However, there is no way for us to provide NLTK this context. What will it be tagged as, will simply be how the word 'visiting' is used in the context of usages in a large corpus. 

    return




######!! PARSING SENTENCE STRUCTURE, CONTEXT FREE GRAMMAR, PROBLEMS
def context_free_grammar():
    text15 = nltk.word_tokenize("Alice loves Bob")
    grammar = nltk.CFG.fromstring("""
    S -> NP VP
    VP -> V NP
    NP -> 'Alice' | 'Bob' | 'you' | 'bear'
    V -> 'loves' | 'bite'
    """)
    # The above is called a context-free grammar
    # Why? Because it doesn't care about context.
    # It will equally accept "I loves me" or "bear bite you"
    # But English is context-sensitive (most languages are)
    # "unfortunately we don't know any efficient algorithm to parse strings of CSG." (stackoverflow)
    # "that's why even in compilers like C-language, we parse CFG first and then later checks context-sensitive features"
    # So this is probably why we need CFG in NLP

    parser = nltk.ChartParser(grammar)
    trees = parser.parse_all(text15)
    for tree in trees:
        print(tree)
    # (S (NP Alice) (VP (V loves) (NP Bob)))

    text16 = nltk.word_tokenize("I saw the man with a telescope")
    grammar1 = nltk.data.load('data/mygrammar.cfg')
    grammar1
    # <Grammar with 13 productions>

    parser = nltk.ChartParser(grammar1)
    trees = parser.parse_all(text16)
    for tree in trees:
        print(tree)
    # (S
    #   (NP I)
    #   (VP
    #     (VP (V saw) (NP (Det the) (N man)))
    #     (PP (P with) (NP (Det a) (N telescope)))))
    # (S
    #   (NP I)
    #   (VP
    #     (V saw)
    #     (NP (Det the) (N man) (PP (P with) (NP (Det a) (N telescope))))))

    #? !pip install svgling # this plots tree graphs
    from nltk.corpus import treebank
    import svgling

    text17 = treebank.parsed_sents('wsj_0001.mrg')[0]
    print(' '.join(sent7))
    # print(text17)
    text17

    ###! Example: Uncommon usages of words (leads to) bad parse
    # in this case, there is only one interpretation, man is a verb. because if man is a noun, the sentence is incomplete. however 

    # from nltk.tokenize import TreebankWordTokenizer
    # treebanktext18 = TreebankWordTokenizer().tokenize("The old man the boat")
    text18 = nltk.word_tokenize("The old man the boat")

    print(  nltk.pos_tag(text18)  )
    # print(  treebanktext18  )
    # [('The', 'DT'), ('old', 'JJ'), ('man', 'NN'), ('the', 'DT'), ('boat', 'NN')]

    ###! Example: Well-formed and correctly parsed sentences may still be meaningless
    text19 = nltk.word_tokenize("Colorless green ideas sleep furiously")
    nltk.pos_tag(text19)
    # [('Colorless', 'NNP'),
    #  ('green', 'JJ'),
    #  ('ideas', 'NNS'),
    #  ('sleep', 'VBP'),
    #  ('furiously', 'RB')]

    # it gets colorless as a proper noun, when actually it should be adjective
    # but even if you removed that word, "Green ideas sleep furiously",
    # it would have correct POS tags, but still no meaning

    ####! Conclusions
    # POS tagging provides insights into the word classes/types in a sentence
    # parsing the grammatical structures help derive meaning
    # both tasks are difficult, linguistic ambiguity increases the difficulty even more
    # we need better models, and they could be learned with supervised training
    # NLTK provides access to tools, as well as some data for training (treebank)
    return




######!! SIMILARITY AND SPELL CHECKING
def spell_checks():
    from nltk.corpus import words

    correct_spellings = words.words()

    def answer_nine(entries=['cormulent', 'incendenece', 'validrate']):
        from nltk.metrics.distance import jaccard_distance
        from nltk.util import ngrams

        suggestion=[]

        for word in entries:
            temp = [ (jaccard_distance( set(ngrams(word, n=3)),
                                        set(ngrams(w,    n=3)) ), w)
                    for w in correct_spellings if w[0]==word[0]]
            suggestion.append(  sorted(temp, key=lambda val:val[0])[0][1]  )
            # print(sorted(temp, key=lambda val:val[0])[0][1] )
        
        return suggestion
        
    def answer_ten(entries=['cormulent', 'incendenece', 'validrate']):
        from nltk.metrics.distance import jaccard_distance
        from nltk.util import ngrams

        suggestion=[]

        for word in entries:
            temp = [ (jaccard_distance( set(ngrams(word, n=4)),
                                        set(ngrams(w,    n=4)) ), w)
                    for w in correct_spellings if w[0]==word[0]]
            # print(sorted(temp, key=lambda val:val[0])[:10])
            suggestion.append(  sorted(temp, key=lambda val:val[0])[0][1]  )
            # print(sorted(temp, key=lambda val:val[0])[0][1] )
        
        return suggestion

    def answer_eleven(entries=['cormulent', 'incendenece', 'validrate']):
        from nltk.metrics.distance  import edit_distance

        suggestion=[]

        for word in entries:
            temp = [(edit_distance(word, w),w) for w in correct_spellings if w[0]==word[0]]
            suggestion.append(  sorted(temp, key = lambda val:val[0])[0][1]  )
            
        return suggestion

    print(answer_nine()  )
    print(answer_ten()   )
    print(answer_eleven())
    # ['corpulent', 'indecence', 'validate']
    # ['cormus', 'incendiary', 'valid']
    # ['corpulent', 'intendence', 'validate']
    return



######!! TEXT CLASSIFICATION (SENTIMENT ANALYSIS), TYPES OF TEXTUAL FEATURES
def text_clf():
    ###! Examples of Text Classification (at different scales)
    #? (paragraph or document level)
    #   - topic identification : is this article about politics, sports or tech?
    #   - spam detection
    #   - sentiment analysis : is this movie review positive or negative?
    #? (word level)
    #   - spelling correction : weather or whether? color or colour?

    ###! Why is textual data unique?
    # - Text data presents a unique set of challenges.
    # - All the information you need is in the text
    # - Features can be pulled out from text at different granularities

    ###! Types of textual features
    # - words
    #   - the most common class of features
    #   - handling commonly-occurring words: stop words ('the')
    #   - normalize: make lowercase or leave as-is? United States or united states?
    #   - stemming/lemmatization: do you want to keep list vs listing separate, or stemmed?
    # - characteristics of words
    #   - capitalization (U.S., White House)
    #   - parts of speech (of words in a sentence)()
    #     - e.g. a misspelt word, but we know it's 'weather' because we have determinant 'the' in front.
    #   - grammatical structure, sentence parsing
    #   - grouping words of similar meaning, semantics
    #     - {buy, purchase, ...} group them together
    #     - {Mr. Ms. Dr. Prof.}
    #     - number (you don't want a feature for 0, a feature for 1.. as long as its any number you just want it to be 1 group.)
    #     - time, date, currency
    # - depending on classification task, features may come from inside words and word sequences.
    #   - bigrams, trigrams, n-grams: "White House" is a single term, not two words.
    #   - character sub-sequences in words: "ing", "ion"

    ###! Additional simple stats features
    # Especially useful for spam texts, negative/positive reviews, etc,
    # churning out simple stats can be useful features
    # e.g. message length (spam messages tend to be long, most person texts are short 'c u later')
    # e.g. message contains how many numbers (spam tends to come with 'whatsapp 81234567' )
    # e.g. how many non isalnum characters

    return



def build_additional_features():

    spam_data = pd.read_csv('spam.csv')
    spam_data['target'] = np.where(spam_data['target']=='spam',1,0)

    def add_feature(X, feature_to_add):
        """
        Returns sparse feature matrix with added feature.
        feature_to_add can also be a list of features.
        """
        from scipy.sparse import csr_matrix, hstack
        return hstack([X, csr_matrix(feature_to_add).T], 'csr')

    def digitlengths(row):
        row['digitlength'] = len(re.findall(r'\d', row['text']))
        return row
    def nonwordlengths(row):
        row['nonwlength'] = len(re.findall(r'\W', row['text']))
        return row

    spam_data = spam_data.apply(digitlengths, axis=1)
    spam_data = spam_data.apply(nonwordlengths, axis=1)


    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import SVC
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import MinMaxScaler
    # scaler = MinMaxScaler()
    # X_train_scaled = scaler.fit_transform(X_train)

    vect = TfidfVectorizer(min_df=5).fit(X_train)
    X_train_vect = vect.transform(X_train)
    X_test_vect  = vect.transform(X_test)

    X_train_txtlengths  = X_train.str.len()
    X_test_txtlengths   = X_test.str.len()
    X_train_nonwords    = X_train.apply(nonwordlengths)
    X_test_nonwords     = X_test.apply(nonwordlengths)
    X_train_digits      = X_train.apply(digitlengths)
    X_test_digits       = X_test.apply(digitlengths)

    X_train_vect = add_feature(X_train_vect, X_train_txtlengths )
    X_test_vect  = add_feature(X_test_vect,  X_test_txtlengths  )
    X_train_vect = add_feature(X_train_vect, X_train_nonwords)
    X_test_vect  = add_feature(X_test_vect,  X_test_nonwords )
    X_train_vect = add_feature(X_train_vect, X_train_digits  )
    X_test_vect  = add_feature(X_test_vect,  X_test_digits   )

    feature_names = list(vect.get_feature_names_out()) #TODO  
    [ feature_names.append(i) for i in ['length_of_doc', 'digit_count', 'non_word_char_count'] ]
    feature_series = pd.Series(lr.coef_[0], index=feature_names)
    feature_series = feature_series.sort_values(ascending=True)

    return



######!! NAIVE BAYES CLASSIFIERS
def naive_bayes_clf():
    # - suppose you are interested in classifying search queries in 3 classes: entertainment/computer science/zoology
    # - most common class is 'entertainment'
    # - suppose the query is 'Python'
    #   - python, snake? (zoology)
    #   - python, programming language? (cs)
    #   - python, in monty python (entertainment)
    # - most common class given "python" is zoology.
    # - suppose the query is 'Python download'
    # - most probable class, given 'python download' is (cs)

    #! NB IS A PROBABILISTIC MODEL
    # - update the likelihood of the class given new information
    # - there are priors for each class. initially, typical queries would most probably be (entertainment). given 'python', most probable is (zoology). given 'python download', most probable is (cs).
    # - **Bayes' Rule**:
    #   - posterior probability = (prior probabilities * likelihood) / evidence
        # $P(y|X) = \frac{P(y) \times P(X|y)}{P(x)} $
        # $P(y=CS|'Python') = \frac{P(y=CS) \times P('Python'|y=CS)}{P('Python')} $

    #! PARAMS
    #? prior probabilities: P(y) for all y in Y
    # - In your training set, count how many labels of each: 
    #   - P(entertainment) = 200/400, 
    #   - P(zoology) = 100/400 
    #   - P(computer science) = 100/400
    #? likelihoods: P(x_i|y) for all features x_i and labels y in Y
    # - Count how many times feature x_i appears in instances labeled as class y. 
    # - If there are p instances of class y, and x_i appears in k of those, 
    #   - P(x_i|y) = k/p
    #   - (e.g. look for the word 'actor' in your corpus for matching 'entertainment')

    # Q: If there are 3 classes (|Y| = 3) and 100 features in X, how many parameters does the Naive Bayes model have?
    # A: |Y| +  2 x |X| +x|Y| = 603
    #    
    # Explanation:
    #   A naïve Bayes classifier has two kinds of parameters:
    #   - $Pr(y)$ for every $y$ in $Y$
    #     - so if $|Y| = 3$, there are **three such parameters**.<br/><br/>
    #   - $Pr(x_i | y)$ for every binary feature $x_i$ in $X$ and $y$ in $Y$
    #     - Specifically, for a particular feature $x_1$, the parameters are 
    #     - $Pr(x_1 = 1 | y)$ and $Pr(x_1 = 0 | y)$ for every $y$. 
    #     - So if $|X| = 100$ binary features and $|Y| = 3$, there are **$(2 \times 100) \times 3 = 600$ such features**
    #   - Hence in all, there are 603 features.
    # 
    # Note: Not all of these features are independent. 
    # In particular, $Pr(x_i = 0 | y) = 1 - Pr(x_i = 1 | y)$, for every $x_i$ and $y$. 
    # So, there are only 300 independent parameters of this kind (as the other 300 parameters are just complements of these). 
    # Similarly, the sum of all prior probabilities $Pr(y)$ should be 1. 
    # This means there are only 2 independent prior probabilities. 
    # In all, for this example, there are 302 independent parameters.

    #! SMOOTHING LIKELIHOODS
    # What happens if $P(x_i|y) = 0$? (Feature $x_i$ never occurs in documents labeled $y$)
    # - If you never saw 'actors' in queries that were labelled entertainment, then that probability would be 0. Which is a problem when it's multiplied to the other probabilities, the overall probability would end up being 0.

    # Instead smooth the parameters using Laplace smoothing/Additive smoothing, i.e:
    # - Add a dummy count of 1. They do not change the Maximum Likelihood Estimations significantly if dataset is large

    # Likelihoods (smoothing)
    #   P(x_i|y) = (k+1) / (p+n)
    #   where n is the number of features (because in all, i have added n words as dummies)

    #! MULTINOMIAL AND BERNOULLI NAIVE BAYES
    # - MNB
    #   - Assumes data follows a multinomial distribution
    #   - Assumes features are independent (tho in practice, this is violated, e.g. if we see White, there is likelihood the following word is House)
    #   - Each feature value is a count (word occurrence counts, TF-IDF weighting, ...)
    # - BNB
    #   - Data follows a multivariate Bernoulli distribution Each feature is binary (word is present / absent)(no counts)

    # Ultimately multinomial is more common for text, but Bernoulli still useful if you just need to check for presence/absence

    from sklearn.naive_bayes import MultinomialNB

    mnb = MultinomialNB().fit(X_train, y_train)
    y_pred = mnb.predict(x_test)
    metrics.f1_score(y_test, y_pred, average='micro')

    return



######!! SUPPORT VECTOR MACHINES
##! TENDS TO BE BEST FOR HIGH DIM TEXT CLASSIFICATION
##! USE NB AS BASELINE BENCHMARK.
def svc_text():
    # - amongst the best for high-dimension text classification, you should use NB as a benchmark but generally SVMs will be what gets you best results
    # - recall: to find a linear boundary, many methods:
    #   - perceptron
    #   - linear discriminative analysis
    #   - linear least squares
    # - however, SVMs are linear classifiers that are **maximum margin classifiers**
    # - SVMs only work for binary classification. thus for multiclass problems, we use either:
    #   - one vs rest approach (multi_class=ovr): (more popular cos it requires fewer models trained)
    #     - for classes (a,b,c,d), train 3 models: 
    #     - a/not a, b/not b, c/not c, d/not d
    #   - one vs one approach:
    #     - for classes (a,b,c,d), train C(n,2) (n choose 2) models. (i.e like in a corr table of 4x4 features, the diagonals don't count, because those are univariate. and then along that diagonal, the other side is just a mirror so you don't need that either. so you only left needing the triangle formed.) so for 4 classes, that's (4^2 - 4)/2 = 6 models. 
    #     - a/b, a/c, a/d, b/c, b/d, c/d
    #     - also don't forget class_weight='balanced' if required
    #     - linear kernels usually work best for text.
    # 
    # ### Conclusion
    # - SVMs tend to be the most accurate classifiers, especially in high-dimensional data
    # - Strong theoretical foundation 
    # - Handles only numeric features
    #   - Convert categorical features to numeric features
    #   - Normalization is needed
    # - Hyperplane hard to interpret


    from sklearn.svm import SVC

    svc = SVC(kernel='linear', C=0.1).fit(X_train, y_train)
    y_pred = svc.predict(x_test)
    metrics.f1_score(y_test, y_pred, average='micro')


    from sklearn import model_selection

    # Either:
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.333, random_state=0)
    # Or:
    y_pred = model_selection.cross_val_predict(svc, X, y, cv=5)
    # on large dataset, it's fairly common to use cv=10, since 10% test set is still large, plus you average across the cross-val splits.
    # in fact it is also common to run cross val multiple times, so that you have reduced variance in your results. (back to stats 101: more samples means less variance)

    return



######!! VARIOUS TEXT CLASSIFIERS IN NLTK
def various_text_classifiers():
    # NLTK has some classifier algorithms:
    # - NaiveBayesClassifier
    # - DecisionTreeClassifier
    # - ConditionalExponentialClassifier
    # - MaxentClassifier
    # - WekaClassifier
    # - SklearnClassifier
    # 
    # - NLTK also uses APIs for sklearn and weka. 
    # - you should check out weka as well. (Weka is a more graphical user interface vs sklearn which is more programmatic)

    from nltk.classify import NaiveBayesClassifier 

    classifier = NaiveBayesClassifier.train(train_set)
    classifier.classify(unlabeled_instance) 
    classifier.classify_many(unlabeled_instances)

    nltk.classify.util.accuracy(classifier, test_set) 
    classifier.labels() 
    classifier.show_most_informative_features()


    from nltk.classify import SklearnClassifier 
    from sklearn.naive_bayes import MultinomialNB 
    from sklearn.svm import SVC

    mnb = SklearnClassifier(MultinomialNB()).train(train_set)
    svc = SklearnClassifier(SVC(),kernel='linear').train(train_set)

    return



######!! CASE STUDY SENTIMENT ANALYSIS
def case_study():
    import pandas as pd
    import numpy as np

    df = pd.read_csv('data/Amazon_Unlocked_Mobile.csv')

    # Sample the data to speed up computation
    # df = df.sample(frac=0.1, random_state=10)

    # Drop missing values
    df.dropna(inplace=True)
    # Remove any 'neutral' ratings equal to 3
    df = df[df['Rating'] != 3]

    # Encode 4s and 5s as 1 (rated positively), Encode 1s and 2s as 0 (rated poorly)
    df['Positively Rated'] = np.where(df['Rating'] > 3, 1, 0)

    # We can see we have imbalanced classes, more positive reviews 
    # (a balanced set would have mean=0.5)
    print(  df['Positively Rated'].mean()  ) # 0.7482686025879323

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(df['Reviews'], 
                                                        df['Positively Rated'], 
                                                        random_state=0)

    print('X_train first entry:\n', X_train.iloc[0])
    print('\n\nX_train shape: ', X_train.shape)
    # X_train first entry:
    #  I bought a BB Black and was deliveried a White BB.Really is not a serious provider...Next time is better to cancel the order.

    # X_train shape:  (231207,)

    ######!! 1 CountVectorizer (bag of words)
    #? We have 23052 text reviews. We'll need to convert them into numeric representation that sklearn can use.
    #? bag-of-words approach is:
    #? simple and commonly used way to represent text for use in ML. 
    #? ignores structure and only counts how often each word occurs
    #? (CountVectorizer allows us to use the bag-of-words approach by converting a collection of text documents into a matrix of token counts.)

    #? CountVectorizer().fit() consists of:
    #? 1. tokenization of the train data and 
    #?    - tokenizes each document by finding all sequences of characters of at least two letters or numbers separated by word boundaries. 
    #?    - converts everything to lowercase and
    #? 2. building of the vocabulary.
    #?    - builds a vocabulary using these tokens. 
     
    #? We can get the vocabulary by using get_feature_names_out()
    #? - This vocabulary is built on any tokens that occurred in the training data. 
    #? - Looking at every 2,000th feature, we can get a small sense of what the 
    #?   vocabulary looks like. 
    #?   We can see it looks pretty messy, including words with numbers
    #?   as well as misspellings. 
    #? - By checking the length of get_feature_names_out(), we can see that we're 
    #?   working with over 53,000 features. (using fraction of dataset, 19000 features)

    from sklearn.feature_extraction.text import CountVectorizer

    # Fit the CountVectorizer to the training data
    vect = CountVectorizer().fit(X_train)

    print(vect.get_feature_names_out()[::2000])

    print('how many features: ', len(vect.get_feature_names_out()))

    ###! 2 TRANSFORM TEXT TO NUMERIC VECTORIZED FORMAT
    # transform the documents in the training data to a document-term matrix
    X_train_vectorized = vect.transform(X_train)
    X_train_vectorized
    # <231207x53216 sparse matrix of type '<class 'numpy.int64'>'
    # 	with 6117776 stored elements in Compressed Sparse Row format>

    ###! 3 TRAIN CLF
    from sklearn.linear_model import LogisticRegression

    # Train the model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vectorized, y_train)

    from sklearn.metrics import roc_auc_score

    # Predict the transformed test documents
    predictions = model.predict(vect.transform(X_test))

    print('AUC: ', roc_auc_score(y_test, predictions))

    ###! CHECK FEATURES AND COEFFICIENTS
    # get the feature names as numpy array
    feature_names = np.array(vect.get_feature_names_out())

    # Sort the coefficients from the model
    sorted_coef_index = model.coef_[0].argsort()

    # Find the 10 smallest and 10 largest coefficients
    # The 10 largest coefficients are being indexed using [:-11:-1] 
    # so the list returned is in order of largest to smallest
    print('Smallest Coefs:\n{}\n'.format(feature_names[sorted_coef_index[:10]]))
    print('Largest Coefs: \n{}'.format(feature_names[sorted_coef_index[:-11:-1]]))
    # Smallest Coefs:
    # ['mony' 'worst' 'false' 'worthless' 'horribly' 'messing' 'unsatisfied'
    #  'blacklist' 'junk' 'garbage']

    # Largest Coefs: 
    # ['excelent' 'excelente' '4eeeks' 'exelente' 'efficient' 'excellent'
    #  'loving' 'pleasantly' 'loves' 'mn8k2ll']


    ######!! TF-IDF (ANOTHER APPROACH) (Term Frequency-Inverse Document Frequency)
    #? Tf–idf allows us to weight terms based on how important they are to a document. 
    #? - High weight is given to terms that appear often in a particular document, but don't appear often in the corpus. 
    #? - Features with low tf–idf are either commonly used across all documents or rarely used and only occur in long documents. 
    #? - Features with high tf–idf are frequently used within specific documents, but rarely used across all documents. 
    #? 
    #? Similar to how we used CountVectorizer, we'll instantiate the tf–idf vectorizer and fit it to our training data. 
    #? 
    #? Because tf–idf vectorizer goes through the same initial process of tokenizing the document, we can expect it to return the same number of features. 
    #? 
    #? However, let's take a look at a few tricks for reducing the number of features that might help improve our model's performance or reduce a refitting. 


    #? CountVectorizor and tf–idf Vectorizor both take an argument, min_df, which allows us to specify a minimum number of documents in which a token needs to appear to become part of the vocabulary. 
    #? 
    #? This helps us remove some words that might appear in only a few and are unlikely to be useful predictors. For example, here we'll pass in min_df
    #? = 5, which will remove any words from our vocabulary that appear
    #? in fewer than five documents. 
    #? 
    #? Looking at the length, we can see we've reduced the number of features by over 35,000 to just under 18,000 features. (using fraction of dataset, 19000 features down to 5400 features)

    from sklearn.feature_extraction.text import TfidfVectorizer

    # Fit the TfidfVectorizer to the training data specifiying a minimum document frequency of 5
    vect = TfidfVectorizer(min_df=5).fit(X_train)

    print('how many features: ', len(vect.get_feature_names_out()))

    X_train_vectorized = vect.transform(X_train)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vectorized, y_train)

    predictions = model.predict(vect.transform(X_test))

    print('AUC: ', roc_auc_score(y_test, predictions))
    # how many features:  17951
    # AUC:  0.9266357077003247

    # Next, when we transform our training data, fit our model, make predictions on the transform test data, and compute the AUC score, we can see we, again, get an AUC of about 0.927. 
    # No improvement in AUC score, but we were able to get the same score using far fewer features. 
    feature_names = np.array(vect.get_feature_names_out())

    sorted_tfidf_index = X_train_vectorized.max(0).toarray()[0].argsort()

    print('Smallest tfidf:\n{}\n'.format(feature_names[sorted_tfidf_index[:10]]))
    print('Largest tfidf: \n{}'.format(feature_names[sorted_tfidf_index[:-11:-1]]))
    print()

    sorted_coef_index = model.coef_[0].argsort()

    print('Smallest Coefs:\n{}\n'.format(feature_names[sorted_coef_index[:10]]))
    print('Largest Coefs: \n{}'.format(feature_names[sorted_coef_index[:-11:-1]]))
    print()

    #? SMALLEST TF-IDF 
    # features either commonly appeared across all reviews or only appeared rarely in very long reviews. 
    # ['commenter' 'pthalo' 'warmness' 'storageso' 'aggregration' '1300'
    #  '625nits' 'a10' 'submarket' 'brawns']

    #? LARGEST TF-IDF
    # words which appeared frequently in a review, but did not appear commonly across all reviews.
    # ['defective' 'batteries' 'gooood' 'epic' 'luis' 'goood' 'basico'
    #  'aceptable' 'problems' 'excellant']

    #? SMALLEST/LARGEST COEFFS
    # we can again see which words our model has connected to negative and positive reviews. 
    #? Smallest ['not' 'worst' 'useless' 'disappointed' 'terrible' 'return' 'waste' 'poor' 'horrible' 'doesn']
    #? Largest  ['love' 'great' 'excellent' 'perfect' 'amazing' 'awesome' 'perfectly' 'easy' 'best' 'loves']

    #? PROBLEM: WORD ORDER IS DISREGARDED
    # These reviews are treated the same by our current model
    print(model.predict(vect.transform(['not an issue, phone is working',
                                        'an issue, phone is not working'])))

    # [0 0] 
    # i.e both interpreted as negative review.
    # to solve this, use n-grams

    ######!! N-GRAMS, BI-GRAMS, TRI-GRAMS
    # Fit the CountVectorizer to the training data specifiying a minimum 
    # document frequency of 5 and extracting 1-grams and 2-grams
    vect = CountVectorizer(min_df=5, ngram_range=(1,2)).fit(X_train)

    X_train_vectorized = vect.transform(X_train)

    print(  len(vect.get_feature_names_out())  )


    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vectorized, y_train)

    predictions = model.predict(vect.transform(X_test))

    print('AUC: ', roc_auc_score(y_test, predictions))

    # Just by adding bigrams, the number of features we have has increased to almost 200,000. (fractioned dataset: 5400 features to 29000 features)
    # And after training our logistic regression model and our new features, looks like by adding bigrams, we were able to improve our AUC score to 0.967. 

    feature_names = np.array(vect.get_feature_names_out())

    sorted_coef_index = model.coef_[0].argsort()

    print('Smallest Coefs:\n{}\n'.format(feature_names[sorted_coef_index[:10]]))
    print('Largest Coefs: \n{}'.format(feature_names[sorted_coef_index[:-11:-1]]))
    print()

    # These reviews are now correctly identified
    print(model.predict(vect.transform(['not an issue, phone is working',
                                        'an issue, phone is not working'])))

    # Smallest Coefs:
    # ['no good' 'junk' 'poor' 'slow' 'worst' 'broken' 'not good' 'terrible'
    #  'defective' 'horrible']

    # Largest Coefs: 
    # ['excellent' 'excelente' 'excelent' 'perfect' 'great' 'love' 'awesome'
    #  'no problems' 'good' 'best']

    # [1 0] # interprets correctly this time.
    return


######!! SEMANTIC TEXT SIMILARITY
def semantic_text_similarity():
    ###? Which pair of words are most similar?
    # - deer, elk
    #    - deer, giraffe
    #    - deer, horse
    #    - deer, mouse
    # How can we quantify such similarity?

    ###? Applications of Text Similarity
    # - grouping similar words into semantic concepts
    # - as a building block in NLU (understanding) tasks
    #   - textual entailment
    #   - paraphrasing

    ###? WordNet
    # - Semantic dictionary of (mostly) English words, interlinked by semantic relations
    # - Includes rich linguistic information
    #   - part of speech, word senses, synonyms, hypernyms/ hyponyms, meronyms, distributional related forms, ...
    # - Machine-readable, freely available

    ###? Semantic Similarity Using WordNet
    #- WordNet organizes information in a hierarchy
    #- Many similarity measures use the hierarchy in some way
    #- Verbs, nouns, adjectives all have separate hierarchies

    ###! PATH SIMILARITY
    # based on lowest common subsumer
    # LCS(deer, elk) = deer         # elk is a type of/child leaf of 'deer'
    # LCS(deer, giraffe) = ruminant # 'ruminant' is parent of both.
    # LCS(deer, horse) = ungulate   # 'ungulate' is 3rd level parent of both

    import nltk
    from nltk.corpus import wordnet as wn

    # give me 'deer', the noun meaning, and the first definition of that.
    deer  = wn.synset('deer.n.01')
    elk   = wn.synset('elk.n.01')
    horse = wn.synset('horse.n.01')

    ###!  PATH SIMILARITY
    print(  deer.path_similarity(elk)    )
    print(  deer.path_similarity(horse) )

    ###!  LIN SIMILARITY
    # use an information criteria to find Lin similarity
    from nltk.corpus import wordnet_ic
    brown_ic = wordnet_ic.ic('ic-brown.dat') # infomation content learnt over a large corpus

    print(  deer.lin_similarity(elk, brown_ic)     )
    print(  deer.lin_similarity(horse, brown_ic)     )

    return



######!! COLLOCATIONS AND DISTRIBUTIONAL SIMILARITY
def collocations_and_distributional_similarity():
    # "You know a word by the company it keeps" [Firth, 1957]
    # Two words that frequently appears in similar contexts are more likely to be semantically related
    #  - The friends MET at a CAFE.
    #  - Ali MET Ray at a PIZZERIA.
    #  - Let’s MEET up near the CINEMA.
    #  - The secret MEETING at the RESTAURANT soon became public.

    ###! Distributional Similarity: useful for context
    # Words before, after, within a small window
    # Parts of speech of words before, after, in a small window
    # Specific syntactic relation to the target word
    # Words in the same sentence, same document, 

    ###! Strength of Association between words
    # - How frequent are these?
    #       - Not similar if two words don’t occur together often
    # - Also important to see how frequent are individual words
    #       - 'the' is very frequent, so high chances it co-occurs often with every other word. How do we deal with that? Use:
    # - Pointwise Mutual Information
    #       - `PMI(w,c) = log [P(w,c) / P(w)P(c)]`
    #       - log of ( P(seeing them together) / (P(seeing A) * P(seeing B))

    import nltk
    from nltk.collocations import *
    from nltk.book import *

    ###!  Use NLTK Collocations and Association measures
    bigram_measures = nltk.collocations.BigramAssocMeasures()

    finder = BigramCollocationFinder.from_words(text1) 
    print(finder.nbest(bigram_measures.pmi, 10) )# give me top 10
    print()
    ###!  finder also has other useful functions, such as frequency filter 
    finder.apply_freq_filter(20)
    print(finder.nbest(bigram_measures.pmi, 10) )# give me top 10
    # restricts to only pairs that at least occur 10 times

    return



######!! TOPIC MODELLING (I.E TEXT CLUSTERING) 
def topic_modelling():
    #? - (GOOD FOR EDA ON TEXT) (What are these documents/tweets about?)
    #? = FEATURE SELECTION TECHNIQUE FOR TEXT CLASSIFIERS OR OTHER TASKS. (e.g. find only highly topic-specific words)

    # What we have: the text collection/corpus
    # in LDA, like in k-means clustering, we set the number of clusters/topics.
    # what we don't know/need to find out:
    # the actual topics
    # topic distribution for each document
    #   e.g. this article covers a bit of sports, a bit of science, a bit of education.

    ###! Latent Dirichlet Allocation
    # (https://www.youtube.com/watch?v=T05t-SqKArY)
    # is a generative model that randomly generates words. most probably, using it to generate a text will produce gibberish.
    # but of course there's a tiny probability it could generate a shakespeare script, or in this case: our original documents.
    # if we several generative models, they generate fake docs 1, 2, 3..., and we can compare with the real document.
    # based on that we can see, one model has a higher probability of generating our original document, so this model has better tuning.
    # after many models, the one with the best settings and highest probabilities is the one whose topics we want.
    # (PLSA is another similar model to LDA)

    #! PARAMS
    # alpha, beta - Dirichlet distributions, for topics and words distributions respectively.
    # theta, phi - Multinomial distributions, for topics and words distributions respectively. 
    # the ones we adjust are alpha and beta, and they give rise to theta and phi

    # how many topics? hard to answer. if you are in a domain e.g. medical and you have the answer beforehand, 
    # that's good. otherwise it's like a clustering problem, trial and error.

    #! STEPS
    # many packages available. gensim, lda..
    # preprocess first: 
    # tokenize, normalize(lowercase), 
    # stopword removal (not just 'is the a' but also frequent words in domain but irrelevant to you. e.g. in patient records 'doctor' and 'patient' may occur a lot but are of low value.)
    # stemming
    # transform into document-term matrix
    # build LDA


    #! doc_set: set of pre-processed text documents
    import gensim
    from gensim import corpora, models

    dictionary = corpora.Dictionary(doc_set)

    corpus = [dictionary.doc2bow(doc) for doc in doc_set]

    ldamodel = gensim.models.ldamodel.LdaModel (corpus, num_topics=4, id2word=dictionary, passes=50) 

    print(ldamodel.print_topics(num_topics=4, num_words=5))

    # dictionary is a mapping between IDs and words

    # corpus is going thru all the documents in the doc_set, 
    # and creating a document-to-bagofwords model. 
    # this is the step that creates the document-term matrix.

    # then you create the lda model, and once that's done, 
    # you can ask it to print the topics.. 
    # give me the top 5 words of these topics.

    #! ldamodel can also be used to find topic distribution of documents (inference)
    topic_dis = ldamodel[new_doc]

    # when you have a new document, you apply the lda model 
    # and it will return the topic distribution, i.e 
    # what was the topic distribution across our 4 topics, for this new document.

    return




######!! INFORMATION EXTRACTION
# Goal: Identify and extract fields of interest from free text

# NAMED ENTITIES
#  - [NEWS]     People, Places, Dates, ... 
#  - [FINANCE]  Money, Companies, ... 
#  - [MEDICINE] Diseases, Drugs, Procedures, ...
#             - Sure, it's easy to regex extract names because John Smith capitalization helps 
#             - However, what about 'lung cancer'?
# RELATIONS
#   - What happened to who, when, where, ...

###! Named Entity Recognition
# NAMED ENTITIES
# Noun phrases that are of specific type and refer to specific individuals, places, organizations, ...

# NAMED ENTITY RECOGNITION
# Technique(s) to identify all mentions of pre-defined named entities in text
#   - BOUNDARY DETECTION:       Identify the mention / phrase
#   - TAGGING/CLASSIFICATION:   Identify the type
#           ('Chicago' could be place, name, album, font..)

###! Examples of Named Entity Recognition Tasks
# The patient is a [63-year-old] [female] with a three-year history
# of ^^=[bilateral [hand] numbness] and ^[occasional weakness].
# Within the past year, these symptoms have progressively gotten worse,
# to encompass also her [feet].
# She had a %[workup] by her [neurologist] and an %[MRI] revealed a 
# ^[C5-6 disc herniation] with ^[cord compression] and a 
# ^[T2 signal change] at that level

# here you can also see that named entities are not exclusive. 
#       'hand' and 'bilateral hand numbness' are both valid entities to capture. 
#       and if you want to capture 'hand' too, what about 'cord' or 'disc', 
#       are those necessary too?
# so what is the level of granularity you require? 
#       maybe you just want to capture disease and treatment, then just % and ^ items suffice 

###! Approaches to identify named entities
# depends on the entities
# well-formatted items can rely on Regex
# for other fields, typically an ML approach 

###! Person, Organization, Location/GPE
# Standard Named Entity Recognition (NER) task in NLP research community 
# typically is a four-class model:
#       - PER           - person
#       - ORG           - organization
#       - LOC/GPE       - location
#       - Other/Outside - (any other class)
#   e.g. "John met Brenda" -- John(PER), met(outside), Brenda(PER)

######!! RELATION EXTRACTION
###  identify relationships between named entities
# "Erbitux helps treat lung cancer"
# [Erbitux] -- [treatment] --> [lung cancer]

######!! CO-REFERENCE RESOLUTION
###  disambiguate mentions and group mentions
# "(Anita) met [Joseph] at the market. [He] surprised (her) with a rose."

######!! QUESTION ANSWERING
###  Given a question, find the most appropriate answer from the text
# What does Erbitux treat?
# Who gave Anita the rose?
### Builds on named entity recog, relation extraction, co-reference resolution

######!! SUMMARY
# Information Extraction is important for natural language understanding and making sense of textual data
# Named Entity Recognition is a key building block to address many advanced NLP tasks
# Named Entity Recognition systems extensively deploy supervised machine learning and text mining techniques discussed in this course