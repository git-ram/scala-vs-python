import Runtime._
import java.io.{BufferedWriter, FileWriter}
import java.security.KeyStore.TrustedCertificateEntry
import java.util.concurrent.TimeUnit
import java.io._

import scala.collection.mutable
import scala.collection.mutable.{ArrayBuffer, ListBuffer}
import scala.concurrent.duration.Duration
import scala.concurrent.{Await, Future}
import scala.math
import scala.util.Random

object KNN {

  type SplitList = ArrayBuffer[ArrayBuffer[Array[Any]]]


  def load_csv(filename: String): ArrayBuffer[Array[Any]] = {
    val rows = ArrayBuffer[Array[Any]]()
    val buffer = io.Source.fromFile(filename)
    for (line <- buffer.getLines()) {
      rows  += line.split(",").map(_.trim)
    }
    buffer.close
    return rows
  }

  def str_column_to_float(dataset: ArrayBuffer[Array[Any]], column: Int): ArrayBuffer[Array[Any]] = {
    for(row <- dataset){
       row(column) = (row(column).asInstanceOf[String].stripMargin).toFloat;
    }
    return dataset
  }

  def str_column_to_int(dataset: ArrayBuffer[Array[Any]], column: Int): ArrayBuffer[Array[Any]] = {
    var class_values = for (row <- dataset) yield row(column)
    var unique = class_values.toSet
    var lookup = new mutable.HashMap[Any, Float]
    for((v,i) <- unique.view.zipWithIndex){
      lookup(v) = i
    }
    for(row <- dataset){
      row(column) = lookup(row(column))
    }
    return dataset
  }

  def is_float(value: String): Boolean = {
    try {
      val num = value.toFloat
      return true
    }
      catch
    {
      case e: NumberFormatException => return false
    }
  }

  def data_normalizer(dataset: ArrayBuffer[Array[Any]]): ArrayBuffer[Array[Any]] = {
    var max_list = new ArrayBuffer[Float]()
    var min_list = new ArrayBuffer[Float]()
    for(i <- 0 until dataset(0).length - 1){
      var col = for(row <- dataset) yield row(i).asInstanceOf[Float]
      min_list += col.min
      max_list += col.max
    }

    for (row <- dataset){
      for (col <- 0 until dataset(0).length - 1){
        row(col) = (row(col).asInstanceOf[Float] - min_list(col))/
                                            (max_list(col) - min_list(col))
      }
    }
    return dataset
  }

  def data_prep(dataset: ArrayBuffer[Array[Any]]): ArrayBuffer[Array[Any]] = {
    var new_dataset = dataset
    for(col <- dataset(0).indices){
      if (is_float(dataset(0)(col).toString)) {
        new_dataset = str_column_to_float(dataset, col)
      }
      else {
        new_dataset = str_column_to_int(dataset, col)
      }
    }
    var final_data = data_normalizer(new_dataset)
    return final_data
  }

  def p_norm_distance(row1:Array[Any], row2: Array[Any], p: Int): Double = {
    var distance = 0.0
    for (i <- 0 until row1.length - 1){
      distance = distance + math.pow((row1(i).asInstanceOf[Float] - row2(i).asInstanceOf[Float]).abs, p)
    }
    return math.pow(distance, (1/p))
  }

  def jaccard_distance(row1: Array[Any], row2: Array[Any], p:Int): Double = {
    var numerator = 0.0
    var x_times_y = 0.0
    var x_sq = 0.0
    var y_sq = 0.0
    for (i <- 0 until row1.length - 1){
      var row1_f = row1(i).asInstanceOf[Float]
      var row2_f = row2(i).asInstanceOf[Float]
      numerator = numerator + math.pow(row1_f - row2_f, 2)
      x_sq = x_sq + math.pow(row1_f, 2)
      y_sq = y_sq + math.pow(row2_f, 2)
      x_times_y = x_times_y + row1_f * row2_f
    }
    return numerator / (x_sq + y_sq - x_times_y)
  }

  def cross_validation_split(dataset: ArrayBuffer[Array[Any]], n_folds: Int):SplitList = {
    var dataset_split = new SplitList
    var dataset_copy = dataset.clone()
    var fold_size = dataset.length / n_folds
    for (i <- 0 until n_folds){
      var fold = new ArrayBuffer[Array[Any]]
      while (fold.length < fold_size){
        var index = Random.nextInt(dataset_copy.length)
        fold += dataset_copy(index)
        dataset_copy -= dataset_copy(index)
      }
      dataset_split += fold
    }
    return dataset_split
  }


  def get_accuracy(y: ArrayBuffer[Any], y_hat: ArrayBuffer[Any]): Double ={
    var total = 0.0
    for (i <- 0 until y.length)
      if (y_hat(i) == y(i)) total = total + 1
    return (total / y.length) * 100
  }

  def get_scores(f_list: SplitList, num_neighbors: Int,
                 distance_method: (Array[Any], Array[Any], Int) => Double, p: Int):ArrayBuffer[Double]={
    var scores = new ArrayBuffer[Double]
    for(fold <- f_list){
      var train_set_full = f_list.clone()
      train_set_full -= fold
      var train_set = train_set_full.flatten
      var test_set = new ArrayBuffer[Array[Any]]
      for (row <- fold){
        test_set += row
      }
      var y_hat = k_nearest_neighbors(train_set, test_set, num_neighbors, distance_method, p)
      var y = for(row <- fold) yield row.last
      var accuracy = get_accuracy(y, y_hat)
      scores += accuracy
    }
  return scores
  }




  // Method to find the most common element of a collection //TODO much work compared to python
  def get_most_common(list: ArrayBuffer[Any]): Any ={
    val map_to_counts = list.groupBy(identity).mapValues(_.size)
    val maxFreq = map_to_counts.maxBy(_._2)._2
    val list_of_commons = map_to_counts.collect{case(e,f) if f == maxFreq => e}.toList
    return list_of_commons(0)  // Return the first common element
  }

  // Make a prediction with neighbors- dist_method is either p_norm_distance or jaccard_distance
  def predict(train: ArrayBuffer[Array[Any]], test_row: Array[Any], num_neighbors: Int,
              dist_method: (Array[Any], Array[Any], Int) => Double, p: Int): Any ={
    var distances = ListBuffer[(Array[Any], Double)]()  //TODO compare space to python
    for (train_row <- train){
      var dist = dist_method(test_row, train_row, p)
      distances += ((train_row, dist))
    }

    // Sort the distances
    var sorted_dist = distances.sortBy(_._2)     //TODO compare sorting to pyhton's
    var neighbors = ArrayBuffer[Array[Any]]()  // TODO ListBuffer vs ArrayBuffer?

    // Get the closest num_neighbors of neighbors
    for (i <- 0 until num_neighbors)
      neighbors += sorted_dist(i)._1

    // Get the labels of the closest neighbors and find the major label
    var neighbors_labels = for(row <- neighbors) yield row.last  //TODO List comprehension vs yield
    var predicted_label = get_most_common(neighbors_labels)
    return predicted_label
  }

  // KNN algorithm
  def k_nearest_neighbors(train: ArrayBuffer[Array[Any]], test: ArrayBuffer[Array[Any]], num_neighbors: Int,
                          dist_method: (Array[Any], Array[Any], Int) => Double, p: Int): ArrayBuffer[Any] ={
    var predictions = new ArrayBuffer[Any]()
    for (row <- test){
      var output = predict(train, row, num_neighbors, dist_method, p)
      predictions += output
    }
    return predictions
  }


  def compute(folds_list: SplitList, num_folds: Int,
              num_neighbor: Int, p_norm_max: Int): Unit ={
    // Variables to store the results
    var result_jaccard = new ArrayBuffer[ArrayBuffer[Any]](3)
    result_jaccard.appendAll(for(i <- 0.to(2)) yield new ArrayBuffer[Any])
    var result_minkowski = new ArrayBuffer[ArrayBuffer[Any]](4)
    result_minkowski.appendAll(for(i <- 0.to(3)) yield new ArrayBuffer[Any])

    // Jaccard method
    var scores = get_scores(folds_list, num_neighbor, jaccard_distance, 0)
    var avg_score = scores.sum / scores.length
    result_jaccard(0) += num_folds
    result_jaccard(1) += num_neighbor
    result_jaccard(2) += avg_score

    // P_norm distance for various p values
    for (p <- 1 until p_norm_max + 1){
      var scores = get_scores(folds_list, num_neighbor, p_norm_distance, p)
      var avg_score = scores.sum / scores.length
      result_minkowski(0) += num_folds
      result_minkowski(1) += num_neighbor
      result_minkowski(2) += p
      result_minkowski(3) += avg_score
    }
    //print("Computation done.\n")
  }


  def runner(filename: String, n_folds: Int, num_neighbors_max: Int,
             p_norm_max: Int, parallel: Boolean):Unit = {
    var p_norm_range = 1 until p_norm_max + 1
    var dataset = load_csv(filename)
    dataset = data_prep(dataset)
    var number_of_neighbors = 1 until (num_neighbors_max + 1)

    // Split dataset into n_folds folds
    var folds_list = cross_validation_split(dataset, n_folds)

    // Evaluate in parallel
    if (parallel == true){
      //println("*** Using multiprocessing ***")
      implicit val ec: scala.concurrent.ExecutionContext = scala.concurrent.ExecutionContext.global
      val futures = for(n <- number_of_neighbors) yield Future{
        compute(folds_list, n_folds, n, p_norm_max)
      }
      futures.map(Await.result(_, Duration.Inf))
    }
    else  // Evaluate normally
    {
      //println("*** Using sequential computation ***")
      // Variables to store the results
      var result_jaccard = new ArrayBuffer[ArrayBuffer[Any]](3)
      result_jaccard.appendAll(for(i <- 0.to(2)) yield new ArrayBuffer[Any])
      var result_minkowski = new ArrayBuffer[ArrayBuffer[Any]](4)
      result_minkowski.appendAll(for(i <- 0.to(3)) yield new ArrayBuffer[Any])

      for (n <- number_of_neighbors){
        // Jaccard method
        var scores = get_scores(folds_list, n, jaccard_distance, 0)
        var avg_score = scores.sum / scores.length
        result_jaccard(0) += n_folds
        result_jaccard(1) += n
        result_jaccard(2) += avg_score

        // P_norm distance for various p values
        for (p <- 1 until p_norm_max + 1){
          var scores = get_scores(folds_list, n, p_norm_distance, p)
          var avg_score = scores.sum / scores.length
          result_minkowski(0) += n_folds
          result_minkowski(1) += n
          result_minkowski(2) += p
          result_minkowski(3) += avg_score
        }
      }
    }
  }



  val NUM_TRIALS = 10
  val DATASET_LIST = ListBuffer("iris.csv")
  val NUM_FOLDS = 5
  val NUM_NEIGHBOR_MAX = 10
  val P_NORM_MAX = 6
  val IS_PARALLEL = true

  def experiment_runner(filename_list: ListBuffer[String], n_trials: Int, parallel: Boolean): Unit ={
    for (data <- filename_list){
      val file = new File("./Reports/Scala_" + data + "_results")
      val bw = new BufferedWriter(new FileWriter(file))
      for (n_n <- 1.to(NUM_NEIGHBOR_MAX)){
        for (p_n <- 1.to(P_NORM_MAX)){
          var trials_time_sum = 0.0
          for (i <- 0.to(n_trials - 1)){
            var start = System.nanoTime()
            runner(data, NUM_FOLDS, n_n, p_n, IS_PARALLEL)
            var finish = System.nanoTime()
            trials_time_sum = trials_time_sum +((finish - start) / math.pow(10,9))
          }
          bw.write("dataset %s - num_neigh_max %d - p_norm_max %d, %f \n"
              .format(data,n_n,p_n, trials_time_sum/n_trials))
        }
      }
      bw.close()
    }
  }

  def main(args:Array[String]){
    println("Available processors: " + Runtime.getRuntime.availableProcessors() )
    experiment_runner(DATASET_LIST, NUM_TRIALS, IS_PARALLEL)
  }
}


