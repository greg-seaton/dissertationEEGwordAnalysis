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

std::vector<int> columnLocations = {3, 55, 108, 160, 213, 266, 319, 371, 424, 477, 530, 583};
std::vector<int> rowLocations = {3, 39, 116, 192, 269, 345, 421, 498};

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
        return {0};
    }
    if (y < 0 || y >= height) {
        std::cerr << "Y value out of bounds!" << std::endl;
        return {0};
    }
    return pixels[height - 1 - y][x];
}

int replaceFile(fs::path filePath) {
    std::string fileName = filePath.stem().string();  // Get the file name without extension
    // std::cout << "Processing file: " << fileName << std::endl;

    std::string filePathStr = filePath.string();
    std::vector<std::vector<Pixel>> pixels;

    int width, height;
    if (!loadBMP(filePathStr, pixels, width, height)) {
        std::cout << "Could not load BMP at " << filePathStr << std::endl;
        return 1;
    }

    // Define the path for the new CSV file (same name, different extension)
    fs::path csvFilePath = filePath;
    csvFilePath.replace_extension(".csv");

    // Delete the old BMP file (if you want to replace it)
    if (fs::exists(filePath)) {
        try {
            fs::remove(filePath);  // Delete the original BMP file
            // std::cout << "Deleted original BMP file: " << filePath << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error deleting BMP file: " << e.what() << std::endl;
            return 1;
        }
    }

    // Write the new CSV file
    std::ofstream outFile(csvFilePath.string());
    if (!outFile) {
        std::cerr << "Failed to open CSV file for writing!" << std::endl;
        return 1;
    }

    // Write pixel data to the CSV file
    for (int row : rowLocations) {
        for (int column : columnLocations) {
            Pixel pixel = getPixel(pixels, column, row, width, height);
            outFile << (int)pixel.red << std::endl;
        }
    }

    // std::cout << "New CSV file created: " << csvFilePath << std::endl;
    return 0;  // Indicate success
}


int main() {
    std::string directoryPath = "../spectrogramDataGrayscaleBig";
    std::vector<std::string> files;

    try {
        if (fs::exists(directoryPath) && fs::is_directory(directoryPath)) {
            for (const auto& entry : fs::recursive_directory_iterator(directoryPath)) {
                if (fs::is_regular_file(entry.status()) && entry.path().extension() == ".bmp") {
                    replaceFile(entry.path());
                }
            }
        } else {
            std::cerr << "Directory does not exist or is not a directory!" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}
