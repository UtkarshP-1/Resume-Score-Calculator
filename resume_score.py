from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity as cs

# store resume content in a variable
resume = input("Enter Resume\n")

# store job description in a variable
job_description = input("Enter Job Description\n")

cv = CountVectorizer()
count_matrix = cv.fit_transform([resume,job_description])

# store cosine similarity output in a variable
similarity_matrix = cs(count_matrix)

# print similarity score in proper format
print(f"{round(similarity_matrix[0][1]*100,2)} %")

