import scala.collection.mutable
import scala.collection.mutable.{ArrayBuffer, ListBuffer}

object KNN {

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

  def main(args:Array[String]){
    var dataset = load_csv("adult_short.csv")
    dataset = data_prep(dataset)
    return
  }
}
