# 1. Load Required Libraries 
pacman::p_load(dplyr, tm, textstem, cld3, stringr, wordcloud, RColorBrewer, 
               ggplot2, syuzhet, topicmodels, sentimentr, tidytext, slam)
# dplyr is for data manipulation, tm for text mining, textstem for lemmatisation, 
# cld3 for language filtering, stringr for string actions
# wordcloud and rcolorbrewer is for word cloud
# sentimentr, syuzhet and ggplot2 is for sentiment analysis and graph plotting
# tidytext is for token-level sentiment scoring


# 2. Import dataset
# Load dataset from the Data folder
shopee_raw <- read.csv("Data/Shopee_reviews.csv", encoding="UTF-8")


# 3. Create working copy to keep original dataset intact
# Selects only the important categories
reviews_v1 <- shopee_raw %>% select(review_text, review_rating, author_app_version) 


# 4. Language filtering to keep only English reviews
reviews_v1$lang <- cld3::detect_language(reviews_v1$review_text)
reviews_v2 <- reviews_v1 %>% filter(lang == "en")
# Additional filter to remove Tagalog, Malay, and Singlish phrases
reviews_v2 <- reviews_v2 %>%
  filter(!str_detect(review_text, "\\b(eh|kasi|lang|naman|opo|po|si|talaga|yung|ng|ung|ako|pa|ka|ang|nya|din|mo|ba|lah|leh|mah|sia)\\b"))
cat("Filtered English-like reviews:", nrow(reviews_v2), "\n")


# 5. Create corpus for text processing
comments_corpus <- VCorpus(VectorSource(reviews_v2$review_text))


# 6. Case standardisation (lower case for all)
comments_v1 <- tm_map(comments_corpus, content_transformer(tolower))


# 7. Remove punctuation, numbers, emojis, and extra whitespaces
comments_v2 <- tm_map(comments_v1, removePunctuation) 
comments_v3 <- tm_map(comments_v2, removeNumbers) 
toSpace <- content_transformer(function(x, pattern) gsub(pattern, " ", x)) 
comments_v4 <- tm_map(comments_v3, toSpace, "[^A-Za-z']")
comments_v5 <- tm_map(comments_v4, stripWhitespace) 


# Convert cleaned corpus back to dataframe for inspection and further filtering
reviews_v3 <- data.frame(
  review_text_clean = sapply(comments_v5, as.character),
  review_rating = reviews_v2$review_rating,
  author_app_version = reviews_v2$author_app_version,
  stringsAsFactors = FALSE
)


# 8. Remove blank and duplicate reviews AFTER cleaning
reviews_v4 <- reviews_v3 %>%
  filter(review_text_clean != "") %>%                       
  distinct(review_text_clean, .keep_all = TRUE)             
cat("Rows remaining after removing blanks and duplicates:", nrow(reviews_v4), "\n")


# 9. Filter out meaningless short reviews (e.g. "good", "ok")
reviews_v5 <- reviews_v4 %>%
  filter(str_count(review_text_clean, "\\w+") > 2)
cat("Remaining reviews after length filtering:", nrow(reviews_v5), "\n")


# 10. Stopword removal and lemmatisation (for sentiment analysis)
# Loading custom stopwords for sentiment analysis
stopwords_sentiment <- scan("Data/stopwordsSA.dat", what = "character", sep = "\n")

# Merge custom stopword list with English stopword list
stopwords_sentiment_final <- c(stopwords("english"), stopwords_sentiment)

# Apply stopword removal to reviews_v5
comments_sentiment <- VCorpus(VectorSource(reviews_v5$review_text_clean))
comments_sentiment <- tm_map(comments_sentiment, removeWords, stopwords_sentiment_final)
comments_sentiment <- tm_map(comments_sentiment, content_transformer(textstem::lemmatize_strings))
comments_sentiment <- tm_map(comments_sentiment, stripWhitespace)

# Convert cleaned corpus back to dataframe for inspection
reviews_sentiment <- data.frame(
  cleaned_text = sapply(comments_sentiment, as.character),
  review_rating = reviews_v5$review_rating,
  author_app_version = reviews_v5$author_app_version,
  stringsAsFactors = FALSE
)

cat("Reviews after stopword removal and lemmatisation:", nrow(reviews_sentiment), "\n")


# 11. Sampling to preserve review rating proportions 
set.seed(123)  # ensures reproducibility of results

cat("Original rating distribution before sampling:\n")
print(prop.table(table(reviews_sentiment$review_rating)))

# Define total sample size 
sample_size <- 100000

# Calculate proportional sample size for each rating
reviews_sample <- reviews_sentiment %>%
  group_by(review_rating) %>%
  sample_n(size = round(sample_size * n() / nrow(reviews_sentiment))) %>%
  ungroup()

# Verify proportions after sampling
cat("Sampled rating distribution:\n")
print(prop.table(table(reviews_sample$review_rating)))


# Sentiment analysis models 

# 1. sentimentr package
# Get sentiment scores (average sentiment per review)
sentimentr_results <- sentiment_by(reviews_sample$cleaned_text)

# Append sentimentr scores to reviews sample
reviews_sentimentr <- cbind(
  reviews_sample[, c("cleaned_text", "review_rating", "author_app_version")],
  sentimentr_results
)

# Preview first 5 reviews with their sentiment score
head(reviews_sentimentr[, c("cleaned_text", "review_rating", 
                            "author_app_version", "word_count", 
                            "ave_sentiment")], 5)

# Visualise overall sentiment polarity distribution
ggplot(reviews_sentimentr, aes(x = ave_sentiment)) +
  geom_histogram(bins = 40, fill = "steelblue", colour = "white", alpha = 0.8) +
  theme_minimal() +
  labs(title = "Distribution of Sentiment Polarity (sentimentr)",
       x = "Average Sentiment Score",
       y = "Count of Reviews")

# create positive/neutral/negative labels 
reviews_sentimentr$pre_sentiment[reviews_sentimentr$ave_sentiment > 0] <- 1   # positive
reviews_sentimentr$pre_sentiment[reviews_sentimentr$ave_sentiment == 0] <- 0  # neutral
reviews_sentimentr$pre_sentiment[reviews_sentimentr$ave_sentiment < 0] <- -1  # negative

# Calculate proportion of pos/neg reviews per rating
sentimentr_rating_summary <- reviews_sentimentr %>%
  group_by(review_rating) %>%
  summarise(
    avg_positive = mean(pre_sentiment == 1),
    avg_negative = mean(pre_sentiment == -1)
  )

print(sentimentr_rating_summary)

# just to try
ggplot(sentimentr_rating_summary, aes(x = factor(review_rating))) +
  geom_col(aes(y = avg_positive), fill = "steelblue", alpha = 0.7) +
  geom_col(aes(y = -avg_negative), fill = "tomato", alpha = 0.7) +
  labs(title = "Positive vs Negative Sentiment by Star Rating (sentimentr)",
       x = "Star Rating",
       y = "Proportion") +
  theme_minimal()

# 2. tidytext (AFINN lexicon)
# Tokenise text and join with AFINN lexicon
afinn <- get_sentiments("afinn")

sentiment_tidy <- reviews_sample %>%
  unnest_tokens(word, cleaned_text) %>%
  inner_join(afinn, by = "word") %>%
  group_by(review_rating) %>%
  summarise(mean_sentiment = mean(value, na.rm = TRUE))

# View sentiment averages by star rating
print(sentiment_tidy)

# visualise sentiment by star rating
ggplot(sentiment_tidy, aes(x = factor(review_rating), y = mean_sentiment, fill = review_rating)) +
  geom_col(show.legend = FALSE) +
  labs(title = "Average Sentiment Score by Review Rating (AFINN Lexicon)",
       x = "Review Rating", y = "Average Sentiment") +
  theme_minimal()

# Step 1: Tokenise and join to AFINN (same as before, but no summarise yet)
tidy_words <- reviews_sample %>%
  mutate(doc_id = row_number()) %>%      # Add review ID
  unnest_tokens(word, cleaned_text) %>%
  inner_join(afinn, by = "word")

# Step 2: Compute mean sentiment per review
tidy_review_scores <- tidy_words %>%
  group_by(doc_id) %>%
  summarise(mean_sentiment = mean(value, na.rm = TRUE)) %>%
  ungroup()

# Step 3: Join back to original text + rating
tidy_review_scores <- tidy_review_scores %>%
  left_join(reviews_sample %>% mutate(doc_id = row_number()),
            by = "doc_id")

# Step 4: Show first 20 reviews with sentiment scores
head(tidy_review_scores[, c("cleaned_text", "review_rating",
                            "author_app_version", "mean_sentiment")], 20)

# 3. syuzhet (NRC lexicon)
# Extract emotion scores using NRC lexicon
syuzhet_scores <- get_nrc_sentiment(reviews_sample$cleaned_text)

# Combine with main dataset sample
reviews_syuzhet <- cbind(reviews_sample, syuzhet_scores)

# 3. Summarise overall sentiment
sentiment_summary <- colSums(syuzhet_scores)
print(sentiment_summary)

# 4. Calculate average positive and negative sentiment by star rating
sentiment_check <- reviews_syuzhet %>%
  group_by(review_rating) %>%
  summarise(
    avg_pos = mean(positive, na.rm = TRUE),
    avg_neg = mean(negative, na.rm = TRUE)
  )

print(sentiment_check)

# 5. Summarise total counts of each NRC sentiment category
sentiment_df <- data.frame(
  sentiment = names(colSums(syuzhet_scores)),
  count = colSums(syuzhet_scores)
)

print(sentiment_df)

# 6. Visualise sentiment distribution
ggplot(sentiment_df, aes(x = reorder(sentiment, -count), y = count, fill = sentiment)) +
  geom_bar(stat = "identity") +
  scale_fill_brewer(palette = "Set3") +  # Supports 12 colours
  scale_y_continuous(labels = scales::comma) +
  labs(title = "Overall Sentiment Distribution in Shopee Reviews",
       x = "Sentiment", y = "Count") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "none")

# show first 20 reviews with sen score
head(reviews_syuzhet[, c("cleaned_text", 
                         "anger", "anticipation", "disgust", "fear", "joy",
                         "sadness", "surprise", "trust",
                         "negative", "positive")], 20)
