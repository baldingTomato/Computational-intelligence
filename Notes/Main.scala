DROP TAKE  #######################################

assert(List(1, 2, 3).drop(2) == List(3))

assert(List(1, 2, 3).take(2) == List(1, 2))


usuwanie zipWithIndex ############################

def remElems[A](seq: Seq[A], k: Int): Seq[A] = {
  seq.zipWithIndex.filter(_._2 != k).map(_._1)
}


foldLeft dla Optionali ###########################

def sumOption(seq: Seq[Option[Double]]): Double = {
  seq.foldLeft(0.0) {
    case (acc, Some(value)) => acc + value
    case (acc, None) => acc
  }
}


foldLeft na Seq ##################################

def deStutter[A](seq: Seq[A]): Seq[A] = {
  seq.foldLeft(Seq.empty[A]) {
    case (acc, x) if acc.isEmpty => acc :+ x
    case (acc, x) if acc.last != x => acc :+ x
    case (acc, x) if acc.last == x => acc
  }

}

acc :+ x -> do Seq acc dokleja na koniec x

def group[A, B, C](l: List[A])(f: A => B)(op: A => C)(op2: (A, C) => C): Set[(B, C)] = {
  @tailrec
  def group_rec(l: List[A], h_l: Set[B] = Set(), acc: Set[(B, C)] = Set()): Set[(B, C)] = l match{
    case Nil => acc
    case head :: tail => {
      (h_l + f(head)).size > h_l.size match {
        case true => return group_rec(tail, h_l + f(head), acc + helper(tail, f(head), (f(head), op(head))))
        case false => return group_rec(tail, h_l, acc)
      }
    }
  }
  def helper(l: List[A], m: B, acc: (B, C)): (B, C) = l match {
    case Nil => acc
    case head :: tail => {
      f(head) == m match {
        case true => {
          acc match {
            case (a, b) => helper(tail, m, (m, op2(head, b)))
          }
        } 
        case false => helper(tail, m, acc)
      }
    }
  }
  group_rec(l)
}

for yield ##########################################

val letters = List('A', 'B')
val numbers = List(1, 2, 3)
val combinations = for {
  l <- letters
  n <- numbers
} yield (l, n)

println(combinations)
// Output: List((A,1), (A,2), (A,3), (B,1), (B,2), (B,3))

val nums = List(1, 2, 3, 4)
val evens = for (n <- nums if n % 2 == 0) yield n
// Equivalent to:
val evens = nums.filter(n => n % 2 == 0)

def threeNumbers(n: Int): Set[(Int, Int, Int)] = {
  (for {
    a <- 1 to n
    b <- a + 1 to n
    c <- b + 1 to n
    if a * a + b * b == c * c
  } yield (a, b, c)).toSet
}

val piniądze = {
  for {
    (numer, liczba) <- lista                    // Iterujemy po wszystkich elementach
    (num, x, jakisY) <- innaLista if num == numer // Dopasowujemy num do listy
  } yield (x, jakisY * liczba)                        // Tworzymy krotkę (x, z)
}
  .groupBy(_._1)                                            // Grupujemy po liczbie x
  .view.mapValues(_.map(_._2).sum)                          // Obliczamy łączny z dla każdej grupy
  .toList                                                   // Konwertujemy mapę na listę
  .sortBy(-_._2)                                            // Sortujemy po z malejąco


// Posortowanie po łącznym z i przypisanie pozycji
  val wyniki = {
    val grouped = piniądze.groupBy(_._2).toList.sortBy(-_._1)     // Grupowanie po z i sortowanie malejąco
    for {
      ((z, items), index) <- grouped.zipWithIndex          // Dodanie indeksów grupom
      (pokoje, _) <- items                                    // Iterowanie po elementach grupy
    } yield (index + 1, pokoje, zysk)                        // Konstrukcja wyników
  }


sliding i exists ##################################

def isOrdered[A](seq: Seq[A])(leq: (A, A) => Boolean): Boolean = {
  !seq.sliding(2).exists {
    case Seq(x, y) => !leq(x, y)
  }
}


groupBy ###########################################

def freq[A](seq: Seq[A]): Set[(A, Int)] = {
  seq.groupBy(identity).view.mapValues(_.size).toSet
}

Inicjalizacja z wartością nieskończoności
Double.PositiveInfinity

minBy, maxBy ######################################

def minMax(seq: Seq[(String, Double)]): Option[(String, String)] = {
  if (seq.isEmpty) None
  else {
    val minUser = seq.minBy(_._2)._1
    val maxUser = seq.maxBy(_._2)._1
    Some((minUser, maxUser))
  }
}


ostatnie laby, distinct, count, intersect ##########


def countChars(str: String): Int = {
  str.distinct.length
}

def swap[A](seq: Seq[A]): Seq[A] = {
  seq.grouped(2).toSeq.flatMap {
    case Seq(x, y) => Seq(y, x)
    case Seq(x)    => Seq(x)
  }
}

def score(code: Seq[Int])(move: Seq[Int]): (Int, Int) = {
  val blacks = (code zip move).count { (c, m) => c == m }
  val whites = code.intersect(move).length - blacks

  (blacks, whites)
}

def histogram(max: Int, text: List[String]): String = {
  // Combine all lines into a single string, filter letters, and convert to lowercase
  val letters = text.mkString("").filter(_.isLetter).map(_.toLower)

  // Create a frequency map for each letter
  val frequencies = letters.groupBy(identity).view.mapValues(_.length).toMap

  // Find the maximum frequency for normalization
  val maxFrequency = if (frequencies.nonEmpty) frequencies.values.max else 1

  // Generate the histogram as a string
  frequencies.toSeq
    .sortBy(_._1) // Sort letters alphabetically
    .map { case (letter, count) =>
      // Calculate the bar length normalized to 'max'
      val barLength =
        ((count.toDouble / maxFrequency) * max).ceil.toInt max 1 // Ensure at least 1 star for non-zero counts
      s"$letter:${"*" * barLength} (${count})" // Append the raw count for verification
    }
    .mkString("\n") // Combine all lines into a single string
}





















