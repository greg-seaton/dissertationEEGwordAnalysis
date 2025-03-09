#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <filesystem>
#include <string>

namespace fs = std::filesystem;

#pragma pack(push, 1)
struct BMPHeader {
    uint16_t fileType;
    uint32_t fileSize;
    uint16_t reserved1;
    uint16_t reserved2;
    uint32_t dataOffset;
};

struct DIBHeader {
    uint32_t headerSize;
    int32_t width;
    int32_t height;
    uint16_t planes;
    uint16_t bitsPerPixel;
    uint32_t compression;
    uint32_t imageSize;
    int32_t xPixelsPerMeter;
    int32_t yPixelsPerMeter;
    uint32_t colorsUsed;
    uint32_t colorsImportant;
};
#pragma pack(pop)

struct Pixel {
    uint8_t blue;
    uint8_t green;
    uint8_t red;

    bool operator==(const Pixel& other) const {
        return red == other.red && green == other.green && blue == other.blue;
    }

    bool operator!=(const Pixel& other) const {
        return !(*this == other);
    }
};

bool loadBMP(const std::string& filename, std::vector<std::vector<Pixel>>& pixels, int& width, int& height) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open BMP file!" << std::endl;
        return false;
    }

    BMPHeader bmpHeader;
    DIBHeader dibHeader;

    file.read(reinterpret_cast<char*>(&bmpHeader), sizeof(BMPHeader));
    file.read(reinterpret_cast<char*>(&dibHeader), sizeof(DIBHeader));

    if (bmpHeader.fileType != 0x4D42) {
        std::cerr << "Not a valid BMP file!" << std::endl;
        return false;
    }

    width = dibHeader.width;
    height = dibHeader.height;
    pixels.resize(height, std::vector<Pixel>(width));

    file.seekg(bmpHeader.dataOffset, std::ios::beg);
    int rowSize = ((width * 3 + 3) / 4) * 4;
    std::vector<uint8_t> rowData(rowSize);
    
    for (int y = 0; y < height; ++y) {
        file.read(reinterpret_cast<char*>(rowData.data()), rowSize);
        for (int x = 0; x < width; ++x) {
            pixels[height - y - 1][x] = { rowData[x * 3], rowData[x * 3 + 1], rowData[x * 3 + 2] };
        }
    }

    file.close();
    return true;
}

Pixel getPixel(const std::vector<std::vector<Pixel>>& pixels, int x, int y, int width, int height) {
    if (x < 0 || x >= width) {
        std::cerr << "X value out of bounds!" << std::endl;
        return {0, 0, 0};
    }
    if (y < 0 || y >= height) {
        std::cerr << "Y value out of bounds!" << std::endl;
        return {0, 0, 0};
    }
    return pixels[height - 1 - y][x];
}

std::pair<std::vector<int>, std::vector<int>> getDeets(const std::string& filename) {
    std::vector<std::vector<Pixel>> pixels;
    int width, height;

    if (!loadBMP(filename, pixels, width, height)) {
        return { {}, {} };  // Return empty vectors if loading fails
    }

    Pixel lastPixel = {0, 0, 0};
    Pixel currentPixel = {0, 0, 0};

    std::vector<int> columnLocations;
    std::vector<int> rowLocations;

    // Add to column locations when the color changes (going across the width)
    for (int i = 1; i < width - 1; i++) {
        currentPixel = getPixel(pixels, i, 1, width, height);
        if (currentPixel != lastPixel) {
            columnLocations.push_back(i);
            lastPixel = currentPixel;
        }
    }

    lastPixel = {0, 0, 0};
    currentPixel = {0, 0, 0};

    // Add to row locations when the color changes (going down the height)
    for (int i = 1; i < height - 1; i++) {
        currentPixel = getPixel(pixels, 1, i, width, height);
        if (currentPixel != lastPixel) {
            rowLocations.push_back(i);
            lastPixel = currentPixel;
        }
    }

    return std::make_pair(columnLocations, rowLocations);
}

int main() {
    std::string directoryPath = "spectrogramData";  // Update this path as necessary
    std::vector<std::string> files;

    try {
        if (fs::exists(directoryPath) && fs::is_directory(directoryPath)) {
            for (const auto& entry : fs::recursive_directory_iterator(directoryPath)) {
                if (fs::is_regular_file(entry.status()) && entry.path().extension() == ".bmp") {
                    files.push_back(entry.path().string());
                }
            }
        } else {
            std::cerr << "Directory does not exist or is not a directory!" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    std::pair<std::vector<int>, std::vector<int>> deets;
    std::vector<std::vector<int>> allColumnLocations;
    std::vector<std::vector<int>> allRowLocations;

    // Process the collected files
    for (const std::string& file : files) {
        deets = getDeets(file);

        // Check if deets.first (columnLocations) is not already in allColumnLocations
        if (std::find(allColumnLocations.begin(), allColumnLocations.end(), deets.first) == allColumnLocations.end()) {
            allColumnLocations.push_back(deets.first);
        }

        // Check if deets.second (rowLocations) is not already in allRowLocations
        if (std::find(allRowLocations.begin(), allRowLocations.end(), deets.second) == allRowLocations.end()) {
            allRowLocations.push_back(deets.second);
        }
    }

    // Optionally output the collected column and row locations
    std::cout << "Unique Column Locations: " << std::endl;
    for (const auto& loc : allColumnLocations) {
        for (int col : loc) {
            std::cout << col << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "Unique Row Locations: " << std::endl;
    for (const auto& loc : allRowLocations) {
        for (int row : loc) {
            std::cout << row << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
