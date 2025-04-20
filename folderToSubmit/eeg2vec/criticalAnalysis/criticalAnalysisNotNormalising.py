import nltk
from nltk import pos_tag
import re
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from collections import defaultdict



def loadCosineSimilarityMatrix(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # Clean the text and extract all float values
    cleaned = content.replace('[', '').replace(']', '').replace('\n', ' ')
    float_strings = cleaned.split()
    float_values = [float(x) for x in float_strings]

    # Infer the matrix size
    total_values = len(float_values)
    size = int(np.sqrt(total_values))

    if size * size != total_values:
        print ("Matrix size error",size)
    matrix = np.array(float_values).reshape((size, size))
    return matrix

def getPOStag(word):
    tag = pos_tag([word])[0][1]
    
    #convert tag to categories
    if tag.startswith('NN'):
        return "noun"
    elif tag.startswith('VB'):
        return "verb"
    elif tag.startswith('JJ'):
        return "adjective"
    elif tag.startswith('RB'):
        return "adverb"
    elif tag.startswith('PRP') or tag == 'WP':
        return "pronoun"
    elif tag == 'DT' or tag == 'PDT':
        return "article"
    elif tag.startswith('IN'):
        return "preposition"
    elif tag.startswith('CC'):
        return "conjunction"
    elif tag.startswith('CD'):
        return "numeral"
    elif tag.startswith('UH'):
        return "interjection"
    else:
        return "other"

def createPOSdict(words):        
    pos_index_dict = {}
    
    #store word index based on POS tag
    for i, word in enumerate(words):
        pos = getPOStag(word)
        
        if pos not in pos_index_dict:
            pos_index_dict[pos] = []
        pos_index_dict[pos].append(i)
    
    return pos_index_dict

def produceConfusionHistogram():
    #x axis cosine similarity
    #y axis quantity
    return

def get_word_pos_tags_by_similarity(word_index, cosineSimilarityMatrix, words):    
    # Get similarities for this word
    similarities = cosineSimilarityMatrix[word_index, :]
    
    # Create a list of (similarity, pos_tag, index, word) tuples
    similarity_pos_list = []
    for i, sim in enumerate(similarities):
        pos = getPOStag(words[i])
        similarity_pos_list.append((sim, pos, i, words[i]))
    
    # Sort by similarity (highest first)
    similarity_pos_list.sort(reverse=True)
    
    return similarity_pos_list

def analyze_pos_by_similarity(pos_dict, cosineSimilarityMatrix, words):
    """
    Analyze similarities for words grouped by their POS tags.
    Run similarity analysis for all nouns, then all verbs, etc.
    """
    # For each POS category
    for pos_category, indices in pos_dict.items():
        print(f"\n\n=== ANALYZING {pos_category.upper()} WORDS ===")

        # Dictionary to store similarity scores by POS
        similarity_by_pos = defaultdict(list)
        words_seen = set()  # To track unique words
        
        # Process each word in this POS category
        for i in indices:
            word = words[i]
            
            # Skip if we've already processed this word
            if word.lower() in words_seen:
                continue
            words_seen.add(word.lower())
            
            word_pos = getPOStag(word)
            print(f"\nWord {i}: '{word}' - POS: {word_pos}")

            # Get POS tags ranked by similarity
            similarity_rankings = get_word_pos_tags_by_similarity(i, cosineSimilarityMatrix, words)
            
            words_already_printed = []
            count = 0
            
            for j, (sim, pos, idx, similar_word) in enumerate(similarity_rankings):
                if idx == i:  # Skip the word itself
                    continue
                
                if similar_word in words_already_printed:
                    continue
                
                # Store similarity score grouped by POS
                similarity_by_pos[pos].append(sim)
                
                print(f"  {count}. Similarity: {sim:.4f} - Word: '{similar_word}' - POS: {pos}")
                count += 1
                words_already_printed.append(similar_word)
        
        # Create histogram for this source POS category
        create_pos_similarity_histogram(pos_category, similarity_by_pos)

def create_pos_similarity_histogram(source_pos, similarity_by_pos):
    """
    Create a stacked histogram showing similarity distributions by POS.
    
    Args:
        source_pos: The source POS category being analyzed
        similarity_by_pos: Dictionary mapping target POS to lists of similarity scores
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Define main POS categories and colors
    main_pos_categories = ['noun', 'verb', 'adjective', 'adverb', 'pronoun', 
                           'preposition', 'conjunction', 'article', 'other']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']
    
    # Filter out POS categories with no data
    pos_with_data = {pos: scores for pos, scores in similarity_by_pos.items() if scores}
    
    # Skip if no data
    if not pos_with_data:
        print(f"No similarity data for {source_pos}")
        return
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Prepare bins for histogram
    bins = np.linspace(0, 1, 21)  # 20 bins from 0 to 1
    
    # Create a stacked histogram
    bottom = np.zeros_like(bins[:-1])
    
    # First plot main categories in a specific order
    for pos, color in zip(main_pos_categories, colors):
        if pos in pos_with_data:
            values, _ = np.histogram(pos_with_data[pos], bins=bins)
            plt.bar(bins[:-1], values, width=0.05, bottom=bottom, alpha=0.7,
                    label=f"{pos} (n={len(pos_with_data[pos])})", color=color)
            bottom += values
            del pos_with_data[pos]  # Remove to avoid plotting twice
    
    # Then plot any remaining categories
    for i, (pos, scores) in enumerate(pos_with_data.items()):
        values, _ = np.histogram(scores, bins=bins)
        plt.bar(bins[:-1], values, width=0.05, bottom=bottom, alpha=0.7,
                label=f"{pos} (n={len(scores)})")
        bottom += values
    
    # Add labels and title
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Similarity Scores for {source_pos.upper()} Words')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.xlim(0, 1)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(f'similarity_histogram_{source_pos}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Histogram for {source_pos} saved as similarity_histogram_{source_pos}.png")
    

words =['Gabriel', 'his30', 'eat', 'back2', 'proud', 'tucked', 'at12', 'office1', 'badly', 'sat4', 'Jake', 'and55', 'was14', 'he2', 'turned2', 'die1', 'The7', 'It', 'is5', 'his43', 'trick', 'the53', 'last1', 'closed2', 'down7', 'stomach', 'fire', 'called2', 'his7', 'out14', 'looking3', 'me3', 'comfort', 'tables', 'and8', 'all', 'right', 'said', 'walked9', 'early1', 'did', 'and47', 'four', 'too2', 'She2', 'of26', 'smiled5', 'calling1', 'that4', 'his53', 'you11', 'underneath', 'nursing', 'took', 'him1', 'racing', 'Anthony', 'heard2', 'to44', 'give2', 'don', 'It2', 'the18', 'shall', 'his33', 'to31', 'The9', 'within1', 'you6', 'him9', 'will2', 'you2', 'own', 'is', 'him', 'to26', 'to36', 'aid', 'Whether', 'A2', 'just8', 'the2', 'We', 'do2', 'a22', 'a36', 'a', 'Jim', 'the78', 'shock', 'bit1', 'perfect', 'should', 'after', 'way3', 'and38', 'of23', 'up13', 'and39', 'of10', 'to1', 'little2', 'Billy1', 'him12', 'men2', 'me1', 'drink', 'back7', 'the14', 'soldier', 'first1', 'the87', 'hands', 'heaved', 'hands2', 'if2', 'big', 'the50', 'I9', 'morning', 'his13', 'slammed', 'on8', 'steady', 'and59', 'go3', 'Jimmy', 'Bob', 'follow2', 'gave3', 'breath1', 'Turn', 'to51', 'else', 'it2', 'smiled1', 'against4', 'took3', 'Roger', 'place', 'trying', 'Men', 'things2', 'me5', 'Sam', 'follow1', 'Jack', 'she8', 'door2', 'that2', 'sandwich', 'the77', 'sighed1', 'door3', 'you1', 'home2', 'breakfast', 'It1', 'his9', 'idea', 'a5', 'Bernard', 'demands', 'bowls', 'to', 'fork', 'After', 'the45', 'pockets', 'enough', 'escape1', 'had6', 'Barbara', 'not1', 'predicted', 'the46', 'axe', 'to2', 'as3', 'walked6', 'was6', 'truck3', 'would', 'Andy1', 'laid', 'his35', 'They', 'a31', 'and71', 'I18', 'he16', 'broken1', 'now1', 'on11', 'one', 'instead', 'was18', 'and12', 'wiped1', 'his6', 'talk2', 'and2', 'tell', 'truck2', 'he18', 'a33', 'and64', 'She7', 'horse', 'but', 'photograph', 'the8', 'the83', 'and15', 'ability', 'be3', 'sniffed', 'still', 'to54', 'stayed', 'mug', 'got', 'I1', 'my2', 'we1', 'Steve', 'Bruce', 'to22', 'light', 'and7', 'chance', 'stomach1', 'but5', 'let2', 'their', 'when4', 'plate', 'to18', 'him2', 'voice', 'with6', 'step1', 'chocolate', 'getting2', 'like4', 'demand', 'The15', 'back6', 'the3', 'of11', 'follow', 'out9', 'He17', 'at5', 'he4', 'almost2', 'heard1', 'dad', 'against5', 'for3', 'to53', 'more3', 'I2', 'when2', 'the58', 'morning4', 'had5', 'mother1', 'Ian', 'this2', 'just2', 'out3', 'trench', 'rather', 'enter', 'The14', 'He19', 'and67', 'again1', 'back9', 'and4']
words = [re.sub(r'\d+$', '', word) for word in words] #remove trailing numbers

cosineSimilarityMatrix=loadCosineSimilarityMatrix("../models/almostFinalEEG2vec/cosine_similarity_matrix.txt")

print ("shape:",cosineSimilarityMatrix.shape)

pos_dict = createPOSdict(words)

# print("POS Tag Index Dictionary:")
# for pos, indices in pos_dict.items():
#     print(f"{pos}: {indices}")

analyze_pos_by_similarity(pos_dict, cosineSimilarityMatrix, words)

