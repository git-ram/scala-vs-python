import scala.collection.mutable.{ArrayBuffer, ListBuffer}

object KNN {

  def load_csv(filename: String): Unit = {
    val rows = ArrayBuffer[Array[String]]()
    val buffer = io.Source.fromFile(filename)
    for (line <- buffer.getLines()) {
      rows  += line.split(",").map(_.trim)
    }
    buffer.close
  }


  def main(args:Array[String]){
    load_csv("iris.csv")
  }
}
