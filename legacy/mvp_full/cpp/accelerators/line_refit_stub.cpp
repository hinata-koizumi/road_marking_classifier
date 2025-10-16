#include <vector>
#include <array>

extern "C" {

struct LineSegment {
    double x0, y0, z0;
    double x1, y1, z1;
};

void RefitLines(const LineSegment* input, int count, LineSegment* output) {
    // Stub: copy-through implementation.
    for (int i = 0; i < count; ++i) {
        output[i] = input[i];
    }
}

}
