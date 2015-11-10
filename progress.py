from __future__ import division

import os
import sys
import time

def progress_bar(index, length):
    progress = index / length

    screen_width = int(os.popen('stty size', 'r').read().split()[1])

    percent_width = 4
    percent_progress = ('%d%%' % (progress * 100)).rjust(percent_width)

    incomplete = 'INCOMPLETE'
    complete = 'COMPLETE!!!'
    message_width = len(max((incomplete, complete), key=len)) + 2
    message = complete if index == length else incomplete

    bar_width = screen_width - percent_width - message_width - 4
    bar_progress = '#' * int(progress * bar_width)
    bar_left = '-' * (bar_width - len(bar_progress))
    bar = '|%s%s|' % (bar_progress, bar_left)

    return '%s %s %s' % (percent_progress, bar, message)
