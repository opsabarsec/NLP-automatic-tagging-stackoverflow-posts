# Natural Language Processing : application for tags prediction

AI - automatic assignment of tags to stackoverflow threads

![popular_tags](/documentation/stack.png)

## 1. The sample data
A selection of 50000 stackoverflow post has been made with a SQL request on the website 

https://data.stackexchange.com/stackoverflow/query/new

using the following code: 

SELECT 
	TOP 50000 ViewCount,
    CreationDate,
    Title,
    Body,
    Tags,
    Score,
    CommentCount,
    AnswerCount,
    FavoriteCount
    
FROM Posts
WHERE 
    FavoriteCount > 10
    AND AnswerCount > 1
    AND Score > 100
ORDER BY Score DESC


## 2. Preprocessing
Data have been cleaned and regularized using specific Python libraries for NLP such as BeautifulSoup and ntlk
A first exploration of most frequent tags and associated keywords has been carried out using simple functions as values_count

Code can be found at the following location
notebooks/Stackoverflowtags-Part1.ipynb


## 3. Tags prediction using unsupervised and supervised ML

Data have been processed, TF-IDF scores calculated and based on those tags could be predicted using unsupervised approaches such as LDA and NMF
A more modern package, YAKE, 
https://repositorio.inesctec.pt/bitstream/123456789/7623/1/P-00N-NF5.pdf

has outperformed them. Best results using using supervised learning have been obtained using an optimized Random Forest alghorithm and Logistic Regression

Code can be found at the following location
notebooks/Stackoverflowtags-Part2.ipynb

Results are summarized below

![tags_results](/documentation/results.png)