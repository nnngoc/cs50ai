import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    tran_dict = {}
    corpus_size = len(corpus.keys())
    num_links = len(corpus[page])

    if num_links == 0:
        # Choose randomly one all of all pages in the corpus with equal probability
        for key in corpus.keys():
            tran_dict[key] = 1 / corpus_size

    else:
        d = damping_factor
        N = corpus_size

        for key in corpus.keys():
            if key in corpus[page]:
                # Choose randomly one of the links with probability d and choose one of all pages randomly
                tran_dict[key] = d / num_links + (1 - d) / N
            else:
                # Choose one of all pages randomly
                tran_dict[key] = (1 - d) / N
    
    return tran_dict

def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    sam_dict = corpus.copy()

    # Set initial counter page is zero
    for i in sam_dict:
        sam_dict[i] = 0
    
    sample_page = random.choice(list(corpus))

    for i in range(n):
        
        # Set page choice in sam_dict
        sam_dict[sample_page] += 1

        # Get list of page, weights from transition model
        trans = transition_model(corpus, sample_page, damping_factor)
        pages = list(trans.keys())
        weights = [trans[j] for j in trans]
        
        #Random choose pages by weights
        sample_page = random.choices(pages, weights, k = 1)[0]

        

    # Caculate the proportion of each page 
    for i in sam_dict:
        sam_dict[i] /= n

    return sam_dict

def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    N = len(corpus)
    cur_rank = {}
    new_rank = {}

    # Assign each page a rank os 1/N
    for page in corpus:
        cur_rank[page] = 1 / N

    # Define a funtion to calculate page rank
    def pagerank(corpus, page, rank, d):
        # pr = pr_1 + d * pr_2
        # Chose a page at random and ended up on the page p with probability 1-d
        pr_1 = (1 - d) / N
        # Followed a link from page i to page p with probability d
        pr_2 = 0

        for page_i in corpus:
            
            num_links = len(corpus[page_i])
            pr_i = rank[page_i]

            # Choose one link for every paga in corpus
            if num_links == 0:
                pr_2 += pr_i / N
            # Choose link in num links
            if page in corpus[page_i]:
                pr_2 += pr_i / num_links
        
        pr = pr_1 + d * pr_2
        return pr     

    while True:
        for page in corpus:
            new_rank[page] = pagerank(corpus, page, cur_rank, damping_factor)
        
        change = max([abs(new_rank[i] - cur_rank[i]) for i in cur_rank])
        # Compare page ranks value change with 0,001
        if change < 0.001:
            break
        else:
            cur_rank = new_rank.copy()
    
    return cur_rank
    
if __name__ == "__main__":
    main()

