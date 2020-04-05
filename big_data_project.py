# Databricks notebook source
# MAGIC %md #Big Data Coursework CSC8101
# MAGIC 
# MAGIC ####Group_EX

# COMMAND ----------

sc

# COMMAND ----------

import pandas as pd
ratingsURL = 'https://csc8101storageblob.blob.core.windows.net/datablobcsc8101/ratings.csv'
ratings = spark.createDataFrame(pd.read_csv(ratingsURL))
ratings = ratings.drop("timestamp").persist()

# COMMAND ----------

# MAGIC %md 
# MAGIC Deleted timestamp column since it has no use for this coursework so we are going to work with the following schema:
# MAGIC 
# MAGIC |-- userId: integer (nullable = true)  
# MAGIC |-- movieId: integer (nullable = true)  
# MAGIC |-- rating: double (nullable = true) 

# COMMAND ----------

# MAGIC %md ##Task 1 : summary statistics from the ratings dataset

# COMMAND ----------

# MAGIC %md the user-item matrix has many missing values since not all the movies were rated by all the users.

# COMMAND ----------

display(ratings.select("userId", "movieId", "rating"))

# COMMAND ----------

# MAGIC %md (1) Average number of ratings per users

# COMMAND ----------

# To calculate the average number of ratings per user, the ratings dataframe is grouped by the userId and the entries per user counted. 
# In the next command the created dataframe with number of ratings per user is then used to calculate the average. 
from pyspark.sql.functions import avg
ratingsPerUser = ratings.select("userId", "rating").groupBy("userId").count()
ratingsPerUser.show(5)

# COMMAND ----------

ratingsPerUser.agg({"count": 'avg'}).show()

# COMMAND ----------

# MAGIC %md (2) Average number of ratings per movie

# COMMAND ----------

# Calculating the average number of ratings per movie is similarly done to the average number of ratings per user, only that the ratings dataframe is this time grouped by the movieId. 
ratingsPerMovie = ratings.select("movieId", "rating").groupBy("movieId").count()
ratingsPerMovie.show(5)

# COMMAND ----------

ratingsPerMovie.agg({"count": 'avg'}).show()

# COMMAND ----------

# MAGIC %md (3) Histogram shows the distribution of movie ratings per user
# MAGIC 
# MAGIC count : number of rating per user
# MAGIC 
# MAGIC Density represent frequency of "count"
# MAGIC 
# MAGIC users have not rated the majority of movies they watched

# COMMAND ----------

# Creating the two dataframes 'ratingsPerUser' and 'ratingsPerMovie' simplify the creation of a histogram showing the distribution of the ratings. 
display(ratings.select("userId", "rating").groupBy("userId").count())

# COMMAND ----------

# MAGIC %md (4) Histogram shows the distribution of movie ratings per movie
# MAGIC 
# MAGIC count : number of rating per movie
# MAGIC 
# MAGIC Density represent frequency of "count" 
# MAGIC 
# MAGIC the majority of movies have not been rated by user they watched

# COMMAND ----------

display(ratingsPerMovie)

# COMMAND ----------

# MAGIC %md (5) the user-item matrix
# MAGIC 
# MAGIC matrix has very many missing values since not all the movies were rated by all the users.

# COMMAND ----------

display(ratings.select("userId", "movieId", "rating"))

# COMMAND ----------

# MAGIC %md the average rating for each movie
# MAGIC The histogram below shows that most ratings fall between 3-4

# COMMAND ----------

display(ratings.select("movieId","rating").groupBy("movieId").mean())

# COMMAND ----------

# MAGIC %md ##Task 2: recommendation model

# COMMAND ----------

# MAGIC %md (1) training a recommender model using the ALS algorithm to predict the rating
# MAGIC 
# MAGIC and measuring the perfomenance, using a RMSE performance metric. RMSE is a measure of how accurately the model predicts the response

# COMMAND ----------

from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from time import time

(trainingData, testData) = ratings.randomSplit([0.75, 0.25])
als = ALS(userCol="userId", itemCol="movieId", ratingCol="rating", nonnegative=True, coldStartStrategy="drop", implicitPrefs=False)

model = als.fit(trainingData)

# Evaluate the model by computing the RMSE on the test data
predictions = model.transform(testData)
evaluator = RegressionEvaluator(metricName="rmse", predictionCol="prediction", labelCol="rating")
rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))

# COMMAND ----------

# MAGIC %md (2) using a GridSearch strategy, we have tuned 2 parameters ( the rank and regParam ) hyperparameters of the model with 3 fold cross validation
# MAGIC 
# MAGIC each with 3 diffirent values and that took 48.64 ​minutes as the algorithm checked 9 combinations

# COMMAND ----------

grid = ParamGridBuilder().addGrid(als.rank, [10,25,40]).addGrid(als.regParam, [.01, .1, .05]).build()

validator = CrossValidator(estimator=als, estimatorParamMaps=grid, evaluator=evaluator, numFolds=5)
start = time()
cvModel = validator.fit(trainingData)
print("Grid Search took %.2f seconds " % (time() - start))

bestModel = cvModel.bestModel
print("Best model with parameters>> ","rank: ", bestModel.rank, "\nregParam: ", bestModel._java_obj.parent().getRegParam())

predictions = bestModel.transform(testData)
rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))

# COMMAND ----------

# MAGIC %md (3) check model overfitting issues
# MAGIC 
# MAGIC First we find the RMSE returned for all combinations
# MAGIC 
# MAGIC then we find the average of all the accuracies
# MAGIC 
# MAGIC after that we find the standard deviation of the data to see degree of variance in the results obtained by the model
# MAGIC 
# MAGIC the standard deviation is extremely low, which means that the model has a very low variance,
# MAGIC 
# MAGIC The model also perform roughly similar on test sets with Root-mean-square error = 0.8203030304627332 
# MAGIC 
# MAGIC So that there is no overfiting

# COMMAND ----------

import statistics
print("All RMSE for all combinations :",cvModel.avgMetrics,"\n")
list(cvModel.avgMetrics)
print("mean : ",statistics.mean(list(cvModel.avgMetrics)),"\n")
print("Standard Divation :", statistics.stdev(list(cvModel.avgMetrics)))


# COMMAND ----------

# MAGIC %md (4) smaller size of parameter grid
# MAGIC 
# MAGIC using a GridSearch strategy, we have tuned 2 parameters ( the rank and regParam ) each with 2 diffirent values  with 3 fold cross validation
# MAGIC 
# MAGIC This took 25.5 minutes as the algorithm checked 4 combinations
# MAGIC 
# MAGIC that is faster by roughly 75% than with 9 combinations

# COMMAND ----------

grid = ParamGridBuilder().addGrid(als.rank, [20,45]).addGrid(als.regParam, [.01, .05]).build()

validator = CrossValidator(estimator=als, estimatorParamMaps=grid, evaluator=evaluator, numFolds=5)
start = time()
cvModel = validator.fit(trainingData)
print("Grid Search took %.2f seconds " % (time() - start))

bestModel = cvModel.bestModel
print("rank: ", bestModel.rank, "\nregParam: ", bestModel._java_obj.parent().getRegParam())

predictions = bestModel.transform(testData)
rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))

# COMMAND ----------

# MAGIC %md (5) tuning one parameter
# MAGIC 
# MAGIC We can tune one parameter but to improve the performance, it is better to test values for other parameters and check the RMSE if it improves or not
# MAGIC 
# MAGIC Lower values of RMSE indicate better fit. 

# COMMAND ----------

grid = ParamGridBuilder().addGrid(als.rank, [10,15,20]).build()

validator = CrossValidator(estimator=als, estimatorParamMaps=grid, evaluator=evaluator, numFolds=3)
start = time()
cvModel = validator.fit(trainingData)
print("Grid Search took %.2f seconds " % (time() - start))

bestModel = cvModel.bestModel
print("rank: ", bestModel.rank)

predictions = bestModel.transform(testData)
rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))

# COMMAND ----------

# MAGIC %md In conclusion, we observe better performance as we increase the rank but we also observe that the model does not scale well for rank roughly greater than 50. For such values the time computation is rather high with no significant improvement on RMSE. On the other hand, the golden ration for the regularization parameter is 0.05.

# COMMAND ----------

# MAGIC %md ##Task 3 : Downsampling ratings so it includes 27,000 unique users

# COMMAND ----------

# MAGIC %md (1) creating a set of users and sampling 27,000

# COMMAND ----------

sampleUsers = ratings.select("userId").distinct().sample(False, 0.5).limit(27000)
sampleUsers.count()

# COMMAND ----------

# MAGIC %md (2) collecting all of their ratings from the ratings dataset.

# COMMAND ----------

from pyspark.sql.functions import when
ratings_small = ratings[ratings["userId"].isin(sampleUsers.toPandas()["userId"].tolist())]
ratings_small.count()

# COMMAND ----------

# MAGIC %md (3) Saving sample dataset as a parquet file

# COMMAND ----------

ratings_small.write.mode("overwrite").parquet("ratings-small.parquet")

# COMMAND ----------

# MAGIC %md ##Task 4: user-user network

# COMMAND ----------

# MAGIC %md (1) join two copies of ratings dataset based on movieId so a row is a record of relationship between two different users rating same movie (including rows of users and theirselves)

# COMMAND ----------

ratings_small = spark.read.parquet("ratings-small.parquet")
df1 = ratings_small.select("movieId", "userId")
df2 = ratings_small.select("movieId", "userId").withColumnRenamed("movieId", "movieId2").withColumnRenamed("userId", "userId2")
UserUserMap = df1.join(df2, df1.movieId == df2.movieId2)

# COMMAND ----------

# MAGIC %md (2) select only userId columns. first column represents user1 second column represents user2 
# MAGIC 
# MAGIC  each user is represented by a node in the graph

# COMMAND ----------

UserUserMap = UserUserMap.select("userId", "userId2")
UserUserMap.show(5)

# COMMAND ----------

UserUserMap.count()

# COMMAND ----------

# MAGIC %md (3) count is  the weight of the edge that represent the number of the same movies that have both users rated

# COMMAND ----------

from pyspark.sql.functions import col, sum
edgesWithDuplicates = UserUserMap.groupBy(UserUserMap.columns).count().filter(UserUserMap.userId != UserUserMap.userId2)
edgesWithDuplicates.orderBy("count", ascending=False).show(5)

# COMMAND ----------

# MAGIC %md Set User 1 to the smaller id and user2 to the larger id in order to remove mirrored edges. 
# MAGIC 
# MAGIC Since the graph we are working with is undirected this is not necessary.

# COMMAND ----------

#Set User 1 to the smaller id and user2 to the larger id in order to remove mirrorred edges
#edgesRDD = edgesWithDuplicates.select("userId", "userId2", "count").rdd.map(lambda x: (min(x["userId"], x["userId2"]), max(x["userId"], x["userId2"]), x["count"]))

# COMMAND ----------

# MAGIC %md As we are using an undirected graph, we do not necessarily need to remove the mirrored edges, hence we can continue using the mirrored edges 

# COMMAND ----------

edges = edgesWithDuplicates.withColumnRenamed("userId", "src").withColumnRenamed("userId2", "dst")

# COMMAND ----------

# MAGIC %md Calculate the average and standard deviation

# COMMAND ----------

average = edges.agg({"count": 'avg'})
average.show()

# COMMAND ----------

from pyspark.sql.functions import stddev
std = edges.select(stddev(col("count")).alias("std")).collect()
print(std[0]["std"])

# COMMAND ----------

# MAGIC %md We use (average + (4 * standard deviation)) as our threshold to drop edges, to create a smaller graph, that can be processed in a decent amount of time.  

# COMMAND ----------

threshold = int(round(average.collect()[0]["avg(count)"] + (4 * std[0]["std"])))
print(threshold) #107

# COMMAND ----------

# MAGIC %md filter the dataset based on the threshold

# COMMAND ----------

edgesFiltered= edges.filter("count>" + str(threshold))
edgesFiltered.count()

# COMMAND ----------

verticesUnfiltered = ratings_small.select("userId").distinct().withColumnRenamed("userId", "id")

# COMMAND ----------

verticesUnfiltered.count()

# COMMAND ----------

# MAGIC %md (5) Generate a user-user network 

# COMMAND ----------

from graphframes import *
graph = GraphFrame(verticesUnfiltered, edgesFiltered)
display(graph.edges)

# COMMAND ----------

# MAGIC %md ##Task 5 : the connected components of the graph

# COMMAND ----------

# MAGIC %md (1) calculate the connected components of the graph

# COMMAND ----------

sc.setCheckpointDir("dbfs:/tmp/groupEX/checkpoints")
connectedComponents = graph.connectedComponents()
display(connectedComponents.orderBy("component", ascending = False))

# COMMAND ----------

# MAGIC %md (2) GroupBy the component to get all nodes within the component and find components that containing the largest number of nodes

# COMMAND ----------

# MAGIC %md
# MAGIC * First Result with Threshold 15 (average rounded down) - 26985 in one Component
# MAGIC * Second Result with Threshold 44 (avg + 1 * standard deviation) - 17454 in one Component, else singles
# MAGIC * Third Result with Threshold 72 (avg + 2 * standard deviation) - 12643in one Component, else singles

# COMMAND ----------

connectedComponents.groupBy("component").count().agg({"count": 'max'}).show()

# COMMAND ----------

sortedCc = connectedComponents.groupBy("component").count().orderBy("count", ascending=False)
maxCC = sortedCc.take(1)[0][0]
print(maxCC)

# COMMAND ----------

# MAGIC %md (3) generate a representation of the subgraph containing the largest component,

# COMMAND ----------

subGraphMax = graph.filterEdges("count>1").filterVertices(graph.vertices["id"].isin(connectedComponents.filter("component="+str(maxCC)).toPandas()["id"].tolist())).dropIsolatedVertices()
display(subGraphMax.vertices)

# COMMAND ----------

display(subGraphMax.edges)

# COMMAND ----------

# MAGIC %md (4) save the subgraph

# COMMAND ----------

subGraphMax.edges.write.mode("overwrite").parquet("subGraphEdges.parquet")
subGraphMax.vertices.write.mode("overwrite").parquet("subGraphVertices.parquet")

# COMMAND ----------

subGraphMax.edges.count()

# COMMAND ----------

# MAGIC %md ##Task 6 
# MAGIC ###Newman Girvan parallel implementation with MapReduce

# COMMAND ----------

vertices = sqlContext.createDataFrame([
  ("a", "Alice", 34),
  ("b", "Bob", 36),
  ("c", "Charlie", 30),
  ("d", "David", 29),
  ("e", "Esther", 32),
  ("f", "Fanny", 36),
  ("g", "Gabby", 60)], ["id", "name", "age"])

edges = sqlContext.createDataFrame([
  ("a", "b", "friend"),
  ("b", "a", "friend"),
  ("a", "c", "friend"),  
  ("c", "a", "friend"),
  ("b", "c", "friend"),
  ("c", "b", "friend"),
  ("b", "d", "friend"),
  ("d", "b", "friend"),
  ("d", "e", "friend"),
  ("e", "d", "friend"),
  ("d", "g", "friend"),
  ("g", "d", "friend"),
  ("e", "f", "friend"),
  ("f", "e", "friend"),
  ("g", "f", "friend"),
  ("f", "g", "friend"),
  ("d", "f", "friend"),
  ("f", "d", "friend")
], ["src", "dst", "relationship"])
vertices = vertices.drop("name","age")

vertices = vertices.drop("name","age")
edges= edges.drop('relationship')

# COMMAND ----------

# MAGIC %md 1- finding shortest paths

# COMMAND ----------

e=edges.rdd.map(lambda  x:(x[0],x[1]))
adjList=e.groupByKey().map(lambda x : (x[0], list(x[1])))
adjDict = adjList.collectAsMap() # make a adjacency dictionary of lists

# make a dictionary of key-sets values.
adjDictSet = {}
for k in adjDict:
  adjDictSet[k] = set(adjDict[k])
# broadcast it so every worker can read it since no write operation will be performed
adjBroad = sc.broadcast(adjDictSet)
def getAdjOf(letter):
    return adjBroad.value[letter] 

# COMMAND ----------

def traverseNode(key,val):
  """
  k = (currentId,sourceId)
  v = (currentId,[sourceId,distance,visited,denominator,pathList])
           currentId   [the nodeId that we are currently traversing]
  arr[0] = sourceId    [the nodeId from which BFS has started]
  arr[1] = distance    [int | Distance between targetId and sourceId]
  arr[2] = visited     [boolean| False if this node has not been expanded otherwise true]9
  arr[3] = pathSum     [int | Number of shortest paths from sourceId to currentId]
  arr[4] = pathList    [List| list of visited nodes to reach k node]
  """
  k = val[0]
  v = val[1]
  src = val[1][0]
  returnRows = []
  if (v[2] == False):
    # set node to visited
    v[2] = True
    # append current visited Node to pathList
    v[4].append(k)
    # emit Row
    returnRows.append((key,val))
    
    # Get the nodes that are k's neighbors but have not been visited before
    for a in (getAdjOf(k) - set(v[4])):
      # emit each new path that can be discoved by visiting each Neighbor
      returnRows.append(((a,src),(a,[v[0],v[1] + 1,False,v[3],v[4].copy()])))
  else:
    # do nothing - emit tuple
    returnRows.append((key,val))
  return (returnRows)

def getLowestDistance(x,y):
  """
  Return the pair with the minimum pathSum thus returning the shortest Path
  If two pairs have the same pathSum merge their pathList and add one
  to the pathSum 
  """
  if (x[1][1] == y[1][1]):
    # no need for deepcopy here we are not going to change it later
    listToCopy = []
    if isinstance(x[1][4][1], list):
      for sublist in x[1][4]:
          listToCopy += [sublist]
      listToCopy += [y[1][4]]
    else:
      listToCopy = [x[1][4]] + [y[1][4]]
    return ((x[0],[x[1][0],x[1][1],x[1][2],x[1][3] + 1,listToCopy]))
  if (x[1][1] > y[1][1]):
    return y
  else:
    return x

# COMMAND ----------

# Perform BFS from every node of the graph.
# Each Iteration explores the graph an extra level till all nodes have been visited.
# Using flatMap because we generate new tuples
BFS = vertices.rdd.flatMap(lambda x:traverseNode((x[0],x[0]),(x[0],[x[0],0,False,1,[]])))
# loop until all nodes are visited
while(BFS.filter(lambda a: a[1][1][2] == False).count() > 0):
  BFS = BFS.flatMap(lambda x:traverseNode(x[0],x[1])) 
# Find the shortest paths
BFS = BFS.reduceByKey(lambda x,y: getLowestDistance(x,y))

# COMMAND ----------

# MAGIC %md 2- Calculating Edge Betweenness 

# COMMAND ----------

def calculateBetwenness(v):
  """
  For each edge calculate its betwenness for the current shortest path (v)
  """
  returnRows = []
  if v[3] > 1: # if more than one available shortest paths exist
    for x in v[4]: # iterate each path in the pathList
      for c,elem in enumerate(x):
        if (c == len(x) - 1):
          break
        # for each edge in the shortest path add its correlating weight.
        # Each edge weight is the result of dividing one by the number of shortest paths
        # that can be explored through that edge
        nextElem = x[c+1]
        # Sort the tuple/key so reduceByKey calculates edgeBetweenness faster
        if (nextElem < elem):
          returnRows.append(((nextElem,elem),1/v[3]))
        else:
          returnRows.append(((elem,nextElem),1/v[3]))
  else:
    for c,y in enumerate(v[4]):
      if (c == len(v[4]) - 1):
        break
      nextElem = v[4][c+1]
      if (nextElem < y):
        returnRows.append(((nextElem,y),1))
      else:
        returnRows.append(((y,nextElem),1))
  return (returnRows)

# COMMAND ----------

edgeValues = BFS.flatMap(lambda x: calculateBetwenness(x[1][1])).reduceByKey(lambda x,y:x+y).map(lambda x:(x[0],x[1]/2))
# Edge betweenness of the graph
edgeValues.collect()

# COMMAND ----------

# MAGIC %md 3- Selecting the Edges to be Removed

# COMMAND ----------

betwenness_values = edgeValues.map(lambda x:x[1])
import statistics
maxBetwennessToDrop = max(betwenness_values.collect()) - 2.5*statistics.stdev(betwenness_values.collect())
edgesToDrop = edgeValues.filter(lambda x: x[1] >= maxBetwennessToDrop).map(lambda x:x[0]).collect()
edgesToDrop

# COMMAND ----------

# MAGIC %md 4- Removing the Edges

# COMMAND ----------

# deletes edges from adjacency list
def deleteEdgesInAdj(listOfEdgesToDrop):
  for x in listOfEdgesToDrop:
    if x[0] in adjDict:
      adjDict[x[0]].remove(x[1])
    if x[1] in adjDict:
      adjDict[x[1]].remove(x[0])
      
def deleteEdges(edge,listOfEdges):
  if (edge in listOfEdges or (edge[1],edge[0]) in listOfEdges):
    return False
  return True   

newEdges = edges
newEdges.rdd.filter(lambda x: deleteEdges(x,edgesToDrop)).collect()

# COMMAND ----------

# MAGIC %md Create new graph after removing edges in Girvan Newman algorithm

# COMMAND ----------

newGraph = GraphFrame(vertices, newEdges)
newConnectedComponents = newGraph.connectedComponents()
newConnectedComponents = newConnectedComponents.groupBy("component").count().orderBy("count", ascending=False)
# create new communities
numOfCommunities = newGraph.labelPropagation(maxIter=5)
numOfCommunities.select("id", "label").show()

# COMMAND ----------

# display the number of different communities
numOfCommunities.select("label").distinct().count()

# COMMAND ----------

display(newGraph.edges)
