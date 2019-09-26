#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np


def overlap(min1, max1, min2, max2):
    min_length = min(max1 - min1, max2 - min2)
    return (max(0, min(max1, max2) - max(min1, min2))) / min_length


def is_inside(min_x, max_x, min_y, max_y, min_x1, max_x1, min_y1, max_y1):
    return min_x + 10 > min_x1 and max_x < max_x1 + 10 and min_y + 10 > min_y1 and max_y < max_y1 + 10


def is_small(min_x, max_x, min_y, max_y):
    height = max_y - min_y
    width = max_x - min_x
    return width < 75 or height < 25


def paragraphs_postprocessing(coordinates):
    for i in range(len(coordinates) - 1):
        rect1 = coordinates[i]
        if rect1 is None:
            continue
        r1min = np.amin(rect1, axis=0)
        r1max = np.amax(rect1, axis=0)
        erased = False

        for j in range(i + 1, len(coordinates)):
            rect2 = coordinates[j]
            if rect2 is None:
                continue
            r2min = np.amin(rect2, axis=0)
            r2max = np.amax(rect2, axis=0)
            if is_small(r1min[0], r1max[0], r1min[1], r1max[1]) or\
                    is_inside(r1min[0], r1max[0], r1min[1], r1max[1], r2min[0], r2max[0], r2min[1], r2max[1]):
                coordinates[i] = None
                erased = True
                break
            elif is_small(r2min[0], r2max[0], r2min[1], r2max[1]) or is_inside(r2min[0], r2max[0], r2min[1], r2max[1], r1min[0], r1max[0], r1min[1], r1max[1]):
                coordinates[j] = None

        if erased:
            continue

    new_coords = [cord for cord in coordinates if cord is not None]
    return new_coords
