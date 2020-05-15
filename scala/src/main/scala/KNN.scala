import scala.collection.mutable
import scala.collection.mutable.{ArrayBuffer, ListBuffer}
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

  def jaccard_distance(row1: Array[Any], row2: Array[Any]): Double = {
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


  def get_accuracy(y: Array[Any], y_hat: Array[Any]): Double ={
    var total = 0.0
    for (i <- 0 until y.length)
      if (y_hat(i) == y(i)) total = total + 1
    return (total / y.length) * 100
  }

  def get_scores(f_list: SplitList, num_neighbors: Int,
                 distance_method: (Array[Any], Array[Any], Seq[Int]) => Double, p: Int*):ArrayBuffer[Double]={
    var scores = new ArrayBuffer[Double]
    for(fold <- f_list){
      var train_set = f_list.clone()
      train_set -= fold
      train_set = train_set.flatten
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

def predict(train: ArrayBuffer[Array[Any]], test_row: Array[Any], num_neighbors: Int,
            dist_method: (Array[Any], Array[Any], Seq[Int]) => Double, p: Int*): Any ={
  var distances = new Array //TODO
}






  def main(args:Array[String]){
    var dataset = load_csv("adult_short.csv")
    dataset = data_prep(dataset)
    //jaccard_distance(dataset(0),dataset(1))
    var split = cross_validation_split(dataset,5)
    return
  }
}
