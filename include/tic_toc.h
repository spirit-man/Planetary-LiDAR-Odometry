#pragma once

#include <ctime>
#include <cstdlib>
#include <chrono>


class TicToc
{
  public:
    TicToc()
    {
        tic();
    }

    void tic()
    {
        start = std::chrono::system_clock::now();
    }

    double toc()
    {
        end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        return elapsed_seconds.count() * 1000;
    }

    double tocAndLog(const std::string &stepName, const std::string &filename)
    {
        double time = toc();
        std::ofstream file(filename, std::ios::app);
        if (file.is_open())
        {
            file << std::fixed << std::setprecision(3) << stepName << ": " << time << " ms" << std::endl;
        }
        file.close();
        return time;
    }

  private:
    std::chrono::time_point<std::chrono::system_clock> start, end;
};
