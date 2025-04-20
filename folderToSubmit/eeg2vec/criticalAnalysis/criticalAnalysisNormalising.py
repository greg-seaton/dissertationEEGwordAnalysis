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

def analyze_pos_by_similarity_with_histogram(pos_dict, cosineSimilarityMatrix, words):
    """
    Analyze similarities for words grouped by POS tags and create normalized histograms
    showing the distribution of similarity scores by POS category.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from collections import defaultdict
    
    # Define which POS categories to include in the analysis
    included_pos_categories = ['noun', 'verb', 'adjective', 'adverb', 'pronoun', 'preposition']
    
    # Check if we should analyze this POS category
    pos_categories_to_analyze = [pos for pos in pos_dict.keys() if pos in included_pos_categories]
    
    # For each included POS category
    for pos_category in pos_categories_to_analyze:
        indices = pos_dict[pos_category]
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
                
                # Only store similarity scores for included POS categories
                if pos in included_pos_categories:
                    similarity_by_pos[pos].append(sim)
                
                print(f"  {count}. Similarity: {sim:.4f} - Word: '{similar_word}' - POS: {pos}")
                count += 1
                words_already_printed.append(similar_word)
        
        # Create histogram for this source POS category
        create_normalized_pos_similarity_histogram(pos_category, similarity_by_pos, included_pos_categories)

def create_normalized_pos_similarity_histogram(source_pos, similarity_by_pos, included_pos_categories):
    """
    Create a normalized stacked histogram showing similarity distributions by POS.
    
    Args:
        source_pos: The source POS category being analyzed
        similarity_by_pos: Dictionary mapping target POS to lists of similarity scores
        included_pos_categories: List of POS categories to include in analysis
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Define colors for the included POS categories
    pos_colors = {
        'noun': '#1f77b4',       # blue
        'verb': '#ff7f0e',       # orange
        'adjective': '#2ca02c',  # green
        'adverb': '#d62728',     # red
        'pronoun': '#9467bd',    # purple
        'preposition': '#8c564b' # brown
    }
    
    # Filter out POS categories with no data
    pos_with_data = {pos: scores for pos, scores in similarity_by_pos.items() 
                    if scores and pos in included_pos_categories}
    
    # Skip if no data
    if not pos_with_data:
        print(f"No similarity data for {source_pos}")
        return
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Prepare bins for histogram
    bins = np.linspace(0, 1, 21)  # 20 bins from 0 to 1
    bin_centers = (bins[:-1] + bins[1:]) / 2  # For plotting
    
    # Calculate histograms for each POS
    histograms = {}
    for pos in included_pos_categories:
        if pos in pos_with_data:
            values, _ = np.histogram(pos_with_data[pos], bins=bins)
            histograms[pos] = values
        else:
            histograms[pos] = np.zeros_like(bins[:-1])
    
    # Calculate the total for each bin to normalize
    totals = np.zeros_like(bins[:-1], dtype=float)
    for pos in histograms:
        totals += histograms[pos]
    
    # Replace zeros with ones to avoid division by zero
    totals = np.where(totals == 0, 1, totals)
    
    # Normalize histograms
    normalized_histograms = {}
    for pos in histograms:
        normalized_histograms[pos] = histograms[pos] / totals
    
    # Create a stacked histogram
    bottom = np.zeros_like(bins[:-1], dtype=float)
    
    # Plot in the order of included_pos_categories
    for pos in included_pos_categories:
        if pos in normalized_histograms:
            values = normalized_histograms[pos]
            plt.bar(bin_centers, values, width=0.045, bottom=bottom, alpha=0.8,
                    label=f"{pos} (n={len(pos_with_data.get(pos, []))})", 
                    color=pos_colors.get(pos))
            bottom += values
    
    # Add labels and title
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Proportion')
    plt.title(f'Normalized Distribution of Similarity Scores for {source_pos.upper()} Words')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.xlim(0, 1)
    plt.ylim(0, 1.05)  # Leave space for potential 100%
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(f'normalized_similarity_histogram_{source_pos}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Normalized histogram for {source_pos} saved as normalized_similarity_histogram_{source_pos}.png")
    

words =['Gabriel', 'his30', 'eat', 'back2', 'proud', 'tucked', 'at12', 'office1', 'badly', 'sat4', 'Jake', 'and55', 'was14', 'he2', 'turned2', 'die1', 'The7', 'It', 'is5', 'his43', 'trick', 'the53', 'last1', 'closed2', 'down7', 'stomach', 'fire', 'called2', 'his7', 'out14', 'looking3', 'me3', 'comfort', 'tables', 'and8', 'all', 'right', 'said', 'walked9', 'early1', 'did', 'and47', 'four', 'too2', 'She2', 'of26', 'smiled5', 'calling1', 'that4', 'his53', 'you11', 'underneath', 'nursing', 'took', 'him1', 'racing', 'Anthony', 'heard2', 'to44', 'give2', 'don', 'It2', 'the18', 'shall', 'his33', 'to31', 'The9', 'within1', 'you6', 'him9', 'will2', 'you2', 'own', 'is', 'him', 'to26', 'to36', 'aid', 'Whether', 'A2', 'just8', 'the2', 'We', 'do2', 'a22', 'a36', 'a', 'Jim', 'the78', 'shock', 'bit1', 'perfect', 'should', 'after', 'way3', 'and38', 'of23', 'up13', 'and39', 'of10', 'to1', 'little2', 'Billy1', 'him12', 'men2', 'me1', 'drink', 'back7', 'the14', 'soldier', 'first1', 'the87', 'hands', 'heaved', 'hands2', 'if2', 'big', 'the50', 'I9', 'morning', 'his13', 'slammed', 'on8', 'steady', 'and59', 'go3', 'Jimmy', 'Bob', 'follow2', 'gave3', 'breath1', 'Turn', 'to51', 'else', 'it2', 'smiled1', 'against4', 'took3', 'Roger', 'place', 'trying', 'Men', 'things2', 'me5', 'Sam', 'follow1', 'Jack', 'she8', 'door2', 'that2', 'sandwich', 'the77', 'sighed1', 'door3', 'you1', 'home2', 'breakfast', 'It1', 'his9', 'idea', 'a5', 'Bernard', 'demands', 'bowls', 'to', 'fork', 'After', 'the45', 'pockets', 'enough', 'escape1', 'had6', 'Barbara', 'not1', 'predicted', 'the46', 'axe', 'to2', 'as3', 'walked6', 'was6', 'truck3', 'would', 'Andy1', 'laid', 'his35', 'They', 'a31', 'and71', 'I18', 'he16', 'broken1', 'now1', 'on11', 'one', 'instead', 'was18', 'and12', 'wiped1', 'his6', 'talk2', 'and2', 'tell', 'truck2', 'he18', 'a33', 'and64', 'She7', 'horse', 'but', 'photograph', 'the8', 'the83', 'and15', 'ability', 'be3', 'sniffed', 'still', 'to54', 'stayed', 'mug', 'got', 'I1', 'my2', 'we1', 'Steve', 'Bruce', 'to22', 'light', 'and7', 'chance', 'stomach1', 'but5', 'let2', 'their', 'when4', 'plate', 'to18', 'him2', 'voice', 'with6', 'step1', 'chocolate', 'getting2', 'like4', 'demand', 'The15', 'back6', 'the3', 'of11', 'follow', 'out9', 'He17', 'at5', 'he4', 'almost2', 'heard1', 'dad', 'against5', 'for3', 'to53', 'more3', 'I2', 'when2', 'the58', 'morning4', 'had5', 'mother1', 'Ian', 'this2', 'just2', 'out3', 'trench', 'rather', 'enter', 'The14', 'He19', 'and67', 'again1', 'back9', 'and4']
words = [re.sub(r'\d+$', '', word) for word in words] #remove trailing numbers

cosineSimilarityMatrix=loadCosineSimilarityMatrix("../models/almostFinalEEG2vec/cosine_similarity_matrix.txt")

print ("shape:",cosineSimilarityMatrix.shape)

pos_dict = createPOSdict(words)

# print("POS Tag Index Dictionary:")
# for pos, indices in pos_dict.items():
#     print(f"{pos}: {indices}")

analyze_pos_by_similarity_with_histogram(pos_dict, cosineSimilarityMatrix, words)

