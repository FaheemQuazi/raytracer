#ifndef _FQRT_UTIL_H_
#define _FQRT_UTIL_H_

#include <stdint.h>
#include <string>
#include <vector>

using namespace std::chrono;
#define TIME_NOW high_resolution_clock::now()
#define TIME_DURATION(s, f) duration_cast<duration<double>>((f) - (s)).count()

namespace fqrt {
    namespace time {
        typedef struct frameTimes_tag {
            double hit;
            double light;
        } frameTimes_t;
        double* FTgetHits(frameTimes_t *ft, int len) {
            double *r = (double*)malloc(sizeof(double) * len);
            for (int i = 0; i < len; i++) {
                r[i] = ft[i].hit;
            }
            return r;
        }
        double* FTgetLights(frameTimes_t *ft, int len) {
            double *r = (double*)malloc(sizeof(double) * len);
            for (int i = 0; i < len; i++) {
                r[i] = ft[i].light;
            }
            return r;
        }
    }
    namespace math {
        template<typename T>
        T average(T* values, int len) {
            T sum = 0;
            for (int i = 0; i < len; i++) {
                sum = sum + values[i];
            }
            return sum / len;
        }
        template<typename T>
        T max(T* values, int len) {
            T m = -INFINITY;
            for (int i = 0; i < len; i++) {
                if (values[i] > m) m = values[i];
            }
            return m;
        }
    }
    namespace files {
        // source: https://gist.github.com/cleoold/088c52efc1f26af5d0ad6d56fcaa2883
        const std::string FileDelimeter = "/";
        std::vector<std::string> SplitPath(std::string str) {
            std::vector<std::string> res;
            size_t pos;
            while ((pos = str.find(FileDelimeter)) != std::string::npos)
            {
                res.push_back(str.substr(0, pos));
                str.erase(0, pos + FileDelimeter.length());
            }
            res.push_back(str);
            return res;
        }
        std::string JoinPath(std::vector<std::string> &strs) {
            if (strs.size() == 0) return "";
            std::vector<char> res;
            for (int i = 0; i < strs.size()-1; ++i)
            {
                for (auto c: strs[i]) res.push_back(c);
                for (auto c: FileDelimeter) res.push_back(c);
            }
            for (auto c: strs[strs.size()-1]) res.push_back(c);
            return std::string{res.begin(), res.end()};
        }
    }
}

#endif