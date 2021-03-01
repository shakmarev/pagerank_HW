import os
import random
import re
import sys
from math import isclose

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
    distr = {}
    n = len(corpus)
    # Choose one of all pages.
    for c in corpus:
        distr.update({c: (1 - damping_factor) / n})
    # Choose one of the links.
    for link in corpus[page]:
        distr.update({link: damping_factor / len(corpus[page]) + (1 - damping_factor) / n})

    return distr


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.
    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    pagerank = {}
    random_page = random.choice(list(corpus.keys()))
    samp = []
    # Iterate through sample and add random pages to samp dict.
    for i in range(n):
        samp.append(random_page)
        tm = transition_model(corpus, random_page, damping_factor)
        random_page = random.choices(list(tm.keys()), list(tm.values()))[0]

    # Count how many of certain page is in samp dict.
    for j in list(set(samp)):
        pagerank.update({j: samp.count(j)/n})

    return pagerank


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.
    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # Assigning each page a rank of 1 / n
    n = len(corpus)
    pagerank = {}
    pr = {c: 1/n for c in corpus}
    # Loop until value changes by less than 0.001
    while True:
        for page in corpus:
            sum = 0
            for p in corpus:
                if page in corpus[p]:
                    sum += pr[p] / len(corpus[p])

            pagerank.update({page: (1 - damping_factor) / n + damping_factor * sum})
        # Quit if value changes by less than 0.001
        if abs(pagerank[page] - pr[page]) <= 0.001:
            break

        pr = pagerank

    return pagerank


if __name__ == "__main__":
    main()
