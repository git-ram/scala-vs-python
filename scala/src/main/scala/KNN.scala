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
    var lookup = new mutable.HashMap[Any, Int]
    for((v,i) <- unique.view.zipWithIndex){
      lookup(v) = i
    }
    for(row <- dataset){
      row(column) = lookup(row(column))
    }
    return dataset
  }




  def main(args:Array[String]){
    var dataset = load_csv("adult_short.csv")
    dataset = str_column_to_float(dataset, 10)
    str_column_to_int(dataset, 1)
    return
  }
}
