#include <exception>
#include <iostream>
#include <string>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

int gliner_cli_main(const std::vector<std::string> & args);

#ifdef _WIN32
int wmain(int argc, wchar_t ** argv) {
    try {
        std::vector<std::string> args;
        for (int i = 0; i < argc; ++i) {
            const int size = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, argv[i], -1, nullptr, 0, nullptr, nullptr);
            if (!size) { std::cerr << "error: Invalid Unicode argument\n"; return 1; }
            std::string value(static_cast<size_t>(size), '\0');
            WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, argv[i], -1, value.data(), size, nullptr, nullptr);
            value.pop_back();
            args.push_back(std::move(value));
        }
        return gliner_cli_main(args);
    } catch (const std::exception & error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
#else
int main(int argc, char ** argv) {
    try {
        return gliner_cli_main(std::vector<std::string>(argv, argv + argc));
    } catch (const std::exception & error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
#endif
