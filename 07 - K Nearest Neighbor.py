"""
    Summery of the session:
        KNN stands for K-Nearest Neighbors.
        KNN is a machine learning algorithm used for classifying data.
            Rather than coming up with a numerical prediction such as a students
            grade or stock price it attempts to classify data into certain categories

            k = the amount of neighbors that our algorithm is looking for
            if k = 3
                we look at the 3 nearest neighbors to our point
                vote --> each neighbor votes for their class
                the highest group wins!
                * k has to be an odd number or there might be a tie!

                in order to find the nearest neighbors:
                    the computer draws a line from or point to other neighbors
                    and calculates the magnitude of each line
                        to calculate the magnitude:
                            point1 = (x1, y1)
                            point2 = (x2, y2)
                            d = sqrt(((x2-x1)^2)+((y2-y1)^2))
                            * if you have more than two dimensions you go as follows:
                                d = sqrt(((x2-x1)^2)+((y2-y1)^2)+((z2-z1)^2)+...)
                if you pick a high number for k your model may chose a bad group which is
                far from the nearest neighbor simply because that group hsa more members than the nearest group!
"""