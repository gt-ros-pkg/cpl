/*
  Copyright (C) 2009 Georgia Institute of Technology

  This library is free software: you can redistribute it and/or modify
  it under the terms of the GNU Lesser Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  This library is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU Lesser Public License for more details.

  You should have received a copy of the GNU General Public License
  and the GNU Lesser Public License along with Man.  If not, see
  <http://www.gnu.org/licenses/>.
*/
#include <time.h>

#ifndef Timer_hpp_DEFINED
#define Timer_hpp_DEFINED
namespace Timer {
    static const long long NANOSECONDS_PER_SECOND = 1000000000;
    static long long nanoTime(void) {
        timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        return ts.tv_sec*NANOSECONDS_PER_SECOND + ts.tv_nsec;
    }

    static void nanoSleepForTime(long long nsecs) {
        timespec ts;
        ts.tv_sec = nsecs / NANOSECONDS_PER_SECOND;
        ts.tv_nsec = ts.tv_sec % NANOSECONDS_PER_SECOND;
        nanosleep(&ts, NULL);
    }
};
#endif // Timer_hpp_DEFINED
