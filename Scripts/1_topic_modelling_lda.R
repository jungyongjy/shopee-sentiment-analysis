# 1. Load Required Libraries 
pacman::p_load(dplyr, tm, textstem, cld3, stringr, wordcloud, RColorBrewer, 
               ggplot2, syuzhet, topicmodels, sentimentr, slam, tidytext)
# dplyr is for data manipulation, tm for text mining, textstem for lemmatisation, 
# cld3 for language filtering, stringr for string actions
# wordcloud and rcolorbrewer is for word cloud
# ggplot2 is for visualisations
# sentimentr, syuzhet and tidytext is for sentiment analysis
# topicmodels is for topic modelling, slam for matrix operations


# 2. Import dataset
txtpath <- file.path("Data/Shopee_reviews.csv")
shopee_raw <- read.csv(txtpath, encoding="UTF-8")


# 3. Create working copy to keep original dataset intact
# Selects only the important categories
reviews_v1 <- shopee_raw %>% select(review_text, review_rating, author_app_version) 


# 4. Language filtering to keep only English reviews
reviews_v1$lang <- cld3::detect_language(reviews_v1$review_text)
# Additional filter to remove Tagalog, Malay, and Singlish phrases
reviews_v1 <- reviews_v1 %>%
  filter(lang == "en") %>%
  filter(!str_detect(review_text, "\\b(eh|kasi|lang|naman|opo|po|si|talaga|yung|ng|ung|ako|pa|ka|ang|nya|din|mo|ba|lah|leh|mah|sia)\\b"))

cat("Filtered English-like reviews:", nrow(reviews_v1), "\n")

# 5. Create corpus for text processing
comments_corpus <- VCorpus(VectorSource(reviews_v1$review_text))


# 6. Case standardisation (lower case for all)
comments_corpus <- tm_map(comments_corpus, content_transformer(tolower))


# 7. Remove punctuation, numbers, emojis, and extra whitespaces
comments_corpus <- tm_map(comments_corpus, removePunctuation)
comments_corpus <- tm_map(comments_corpus, removeNumbers)
toSpace <- content_transformer(function(x, pattern) gsub(pattern, " ", x))
comments_corpus <- tm_map(comments_corpus, toSpace, "[^A-Za-z']")
comments_corpus <- tm_map(comments_corpus, stripWhitespace)


# Convert cleaned corpus back to dataframe for inspection and further filtering
reviews_v1 <- data.frame(
  review_text_clean = sapply(comments_corpus, as.character),
  review_rating = reviews_v1$review_rating,
  author_app_version = reviews_v1$author_app_version,
  stringsAsFactors = FALSE
)


# 8 & 9. Remove blank and duplicate reviews AFTER cleaning
# Filter out meaningless short reviews (e.g. "good", "ok")
reviews_v1 <- reviews_v1 %>% 
  filter(review_text_clean != "") %>%
  distinct(review_text_clean, .keep_all = TRUE) %>%
  filter(str_count(review_text_clean, "\\w+") > 2)
cat("Remaining reviews:", nrow(reviews_v1), "\n")

# 10. Term frequency matrix before removing stopwords
set.seed(123)
reviews_sample <- reviews_v1 %>% sample_n(10000) 
dtm_pre <- DocumentTermMatrix(VCorpus(VectorSource(reviews_sample$review_text_clean)))
freq <- sort(colSums(as.matrix(dtm_pre)), decreasing = TRUE)
freq_table <- data.frame(term = names(freq), frequency = freq)
print(head(freq_table, 20))

# 11. Word cloud visualisation to show most common words
set.seed(123)
wordcloud(words = freq_table$term,
          freq = freq_table$frequency,
          min.freq = 20,
          max.words = 100,
          random.order = FALSE,
          rot.per = 0.25,
          colors = brewer.pal(8, "Dark2"))


# 12. Stopword removal and lemmatisation
# Loading custom stopwords
stopwords_topic <- scan("Data/stopwordstopic.dat", what = "character", sep = "\n")

# Merge custom stopword list with English stopword list
stopwords_topic_final <- c(stopwords("english"), stopwords_topic)

# Apply stopword removal to reviews_v1
comments_topic <- VCorpus(VectorSource(reviews_v1$review_text_clean))
comments_topic <- tm_map(comments_topic, removeWords, stopwords_topic_final)
comments_topic <- tm_map(comments_topic, content_transformer(textstem::lemmatize_strings))
comments_topic <- tm_map(comments_topic, stripWhitespace)

# Convert cleaned corpus back to data frame for inspection
reviews_topic <- data.frame(
  cleaned_text = sapply(comments_topic, as.character),
  review_rating = reviews_v1$review_rating,
  author_app_version = reviews_v1$author_app_version,
  stringsAsFactors = FALSE
)

cat("Reviews after stopword removal and lemmatisation:", nrow(reviews_topic), "\n")

# Use same sample as pre-stopword removal to convert to dtm
set.seed(123)
reviews_topic_sample <- reviews_topic[rownames(reviews_sample), ]

# Create DTM using the same 10k cleaned reviews
dtm_post <- DocumentTermMatrix(VCorpus(VectorSource(reviews_topic_sample$cleaned_text)))

# Calculate term frequencies of post-stopword removal sample dtm 
freq_post <- sort(colSums(as.matrix(dtm_post)), decreasing = TRUE)
freq_table_post <- data.frame(term = names(freq_post), frequency = freq_post)

# Preview top 20 most frequent words AFTER stopword removal
print(head(freq_table_post, 20))

# 14. Word cloud post stopword removal using same sample
set.seed(123)
wordcloud(words = freq_table_post$term,
          freq = freq_table_post$frequency,
          min.freq = 10,
          max.words = 100,
          random.order = FALSE,
          rot.per = 0.25,
          colors = brewer.pal(8, "Dark2"))


# 15. Sampling to preserve review rating proportions 
set.seed(123)  # ensures reproducibility of results

cat("Original rating distribution before sampling:\n")
print(prop.table(table(reviews_topic$review_rating)))

# Define total sample size 
sample_size <- 100000

# Calculate proportional sample size for each rating
reviews_sample <- reviews_topic %>%
  group_by(review_rating) %>%
  sample_n(size = round(sample_size * n() / nrow(reviews_topic))) %>%
  ungroup()

# Verify proportions after sampling
cat("Sampled rating distribution:\n")
print(prop.table(table(reviews_sample$review_rating)))


# 16. Create Document-Term Matrix for topic modelling
set.seed(123)
comments_topic_sample <- VCorpus(VectorSource(reviews_sample$cleaned_text))
dtm_post <- DocumentTermMatrix(comments_topic_sample)

# Remove empty documents if any
row_sums <- slam::row_sums(as.matrix(dtm_post)) # to remove rows with 0 words after removing stopwords
dtm_final <- dtm_post[row_sums > 0, ]

# Align reviews to the non-empty documents
valid_docs <- which(row_sums > 0)
reviews_topic_sample <- reviews_sample[valid_docs, , drop = FALSE]

cat("Final rows in dtm_final:", nrow(dtm_final), "\n")
cat("Final rows in reviews_topic_sample:", nrow(reviews_topic_sample), "\n")


# 17. Determine optimal number of topics (k) using silhouette method
# Install and load the required package
install.packages("factoextra")
library(factoextra)

# Create a smaller sample of the DTM for computation efficiency (silhouette is heavy)
# Convert text into a term-document matrix for clustering
set.seed(123)
comments_topic_sample_small <- VCorpus(VectorSource(reviews_sample$cleaned_text))
dtm_small <- DocumentTermMatrix(comments_topic_sample_small)

# Convert DTM to matrix and compute distance matrix
dtm_matrix <- as.matrix(dtm_small)
dtm_dist <- dist(scale(dtm_matrix))

# Perform hierarchical clustering
hc_model <- hclust(dtm_dist, method = "ward.D2")

# Visualise optimal cluster number using silhouette method
fviz_nbclust(dtm_matrix, FUN = hcut, method = "silhouette") +
  labs(title = "Elbow Plot of Silhouette Scores for Optimal k",
       x = "Number of Clusters (k)",
       y = "Average Silhouette Width") +
  theme_minimal()

# 1. Modelling
# 2. Build LDA models
# k = number of topics. Can be adjusted for trial and error
k <- 8
seed <- 123 # for reproducibility
model_lda<-
  list(VEM=LDA(dtm_final,k=k,method="VEM",control=list(seed=seed)),
       Gibbs=LDA(dtm_final,k=k,method="Gibbs",control=list(seed=seed,burnin
                                                           =1000,thin=100,iter=1000)))
# shows top 10 words per topic
terms_vem<-terms(model_lda$VEM,10)
terms_gibbs<-terms(model_lda$Gibbs,10)
terms_vem
terms_gibbs

# Compute perplexity for VEM and Gibbs models
# k = 5
set.seed(123)
lda_vem_5 <- LDA(dtm_final, k = 5, method = "VEM", control = list(seed = 123))
lda_gibbs_5 <- LDA(dtm_final, k = 5, method = "Gibbs",
                   control = list(seed = 123, burnin = 1000, thin = 100, iter = 1000))

perplexity_vem_5 <- perplexity(lda_vem_5, dtm_final)
perplexity_gibbs_5 <- perplexity(lda_gibbs_5, dtm_final)


# k = 8
set.seed(123)
lda_vem_8 <- LDA(dtm_final, k = 8, method = "VEM", control = list(seed = 123))
lda_gibbs_8 <- LDA(dtm_final, k = 8, method = "Gibbs",
                   control = list(seed = 123, burnin = 1000, thin = 100, iter = 1000))

perplexity_vem_8 <- perplexity(lda_vem_8, dtm_final)
perplexity_gibbs_8 <- perplexity(lda_gibbs_8, dtm_final)


# k = 10
set.seed(123)
lda_vem_10 <- LDA(dtm_final, k = 10, method = "VEM", control = list(seed = 123))
lda_gibbs_10 <- LDA(dtm_final, k = 10, method = "Gibbs",
                    control = list(seed = 123, burnin = 1000, thin = 100, iter = 1000))

perplexity_vem_10 <- perplexity(lda_vem_10, dtm_final)
perplexity_gibbs_10 <- perplexity(lda_gibbs_10, dtm_final)

# Combine results into dataframe for plotting
perplexity_vem <- data.frame(
  k = c(5, 8, 10),
  perplexity = c(perplexity_vem_5, perplexity_vem_8, perplexity_vem_10)
)

perplexity_gibbs <- data.frame(
  k = c(5, 8, 10),
  perplexity = c(perplexity_gibbs_5, perplexity_gibbs_8, perplexity_gibbs_10)
)

print(perplexity_vem)
print(perplexity_gibbs)

# Visualise perplexity trend across k-values
# VEM perplexity plot
ggplot(perplexity_vem, aes(x = k, y = perplexity)) +
  geom_line(colour = "steelblue", size = 1.2) +
  geom_point(colour = "steelblue", size = 3) +
  labs(title = "Perplexity Scores for VEM Model",
       x = "Number of Topics (k)",
       y = "Perplexity Score") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))

# Gibbs perplexity plot
ggplot(perplexity_gibbs, aes(x = k, y = perplexity)) +
  geom_line(colour = "darkorange", size = 1.2) +
  geom_point(colour = "darkorange", size = 3) +
  labs(title = "Perplexity Scores for Gibbs Model",
       x = "Number of Topics (k)",
       y = "Perplexity Score") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))

# Extract topic assignments using Gibbs 
topic_gibbs <- topics(model_lda$Gibbs, 1)   # Using Gibbs method 
topic_gibbs[1:5]                            # Display first 5 document-topic mappings

# View topic distribution across documents
topic_gibbs <- topics(model_lda$Gibbs, 2)   # Retrieve probability matrix
topic_gibbs[, 1:5]                          # Show first 5 columns (topics)

# Create a dataframe of topic probabilities (gamma values)
gammaDF_gibbs <- as.data.frame(model_lda$Gibbs@gamma, 
                               row.names = model_lda$Gibbs@documents)
names(gammaDF_gibbs) <- c(1:k)              # Rename columns as topic IDs

# View topic probability table
View(gammaDF_gibbs)

# Retrieve top 5 reviews for Topic 1
gammaDF_gibbs$doc_id <- 1:nrow(gammaDF_gibbs)
gammaDF_gibbs$text <- reviews_topic_sample$cleaned_text
top_topic1 <- gammaDF_gibbs[order(gammaDF_gibbs$`1`, decreasing = TRUE), ]
head(top_topic1$text, 5)

# Retrieve top 5 reviews for Topic 2
gammaDF_gibbs$doc_id <- 1:nrow(gammaDF_gibbs)
gammaDF_gibbs$text <- reviews_topic_sample$cleaned_text
top_topic2 <- gammaDF_gibbs[order(gammaDF_gibbs$`2`, decreasing = TRUE), ]
head(top_topic2$text, 5)

# Retrieve top 5 reviews for Topic 3
gammaDF_gibbs$doc_id <- 1:nrow(gammaDF_gibbs)
gammaDF_gibbs$text <- reviews_topic_sample$cleaned_text
top_topic3 <- gammaDF_gibbs[order(gammaDF_gibbs$`3`, decreasing = TRUE), ]
head(top_topic3$text, 5)

# Retrieve top 5 reviews for Topic 4
gammaDF_gibbs$doc_id <- 1:nrow(gammaDF_gibbs)
gammaDF_gibbs$text <- reviews_topic_sample$cleaned_text
top_topic4 <- gammaDF_gibbs[order(gammaDF_gibbs$`4`, decreasing = TRUE), ]
head(top_topic4$text, 5)

# Retrieve top 5 reviews for Topic 5
gammaDF_gibbs$doc_id <- 1:nrow(gammaDF_gibbs)
gammaDF_gibbs$text <- reviews_topic_sample$cleaned_text
top_topic5 <- gammaDF_gibbs[order(gammaDF_gibbs$`5`, decreasing = TRUE), ]
head(top_topic5$text, 5)

# Retrieve top 5 reviews for Topic 6
gammaDF_gibbs$doc_id <- 1:nrow(gammaDF_gibbs)
gammaDF_gibbs$text <- reviews_topic_sample$cleaned_text
top_topic6 <- gammaDF_gibbs[order(gammaDF_gibbs$`6`, decreasing = TRUE), ]
head(top_topic6$text, 5)

# Retrieve top 5 reviews for Topic 7
gammaDF_gibbs$doc_id <- 1:nrow(gammaDF_gibbs)
gammaDF_gibbs$text <- reviews_topic_sample$cleaned_text
top_topic7 <- gammaDF_gibbs[order(gammaDF_gibbs$`7`, decreasing = TRUE), ]
head(top_topic7$text, 5)

# Retrieve top 5 reviews for Topic 8
gammaDF_gibbs$doc_id <- 1:nrow(gammaDF_gibbs)
gammaDF_gibbs$text <- reviews_topic_sample$cleaned_text
top_topic8<- gammaDF_gibbs[order(gammaDF_gibbs$`8`, decreasing = TRUE), ]
head(top_topic8$text, 5)

# visualising topic distribution by star rating level
topic_assignments <- topics(model_lda$Gibbs)

reviews_topics <- data.frame(
  review_rating = reviews_topic_sample$review_rating,
  topic = topic_assignments
)

# stacked bar chart
ggplot(reviews_topics, aes(x = factor(topic), fill = factor(review_rating))) +
  geom_bar(position = "fill") +
  labs(title = "Topic Distribution by Star Rating",
       x = "Topic", y = "Proportion of Reviews",
       fill = "Star Rating") +
  theme_minimal()

