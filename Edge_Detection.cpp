#include <unistd.h>
#include <string.h>
#include <sys/wait.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <filesystem>
#include <vector>
#include <utility>
#include <tuple>
#include <unordered_set>
#include <initializer_list>
#include <limits>
#include <type_traits>
#include <numeric>
#include <functional>

using namespace std;


template<typename T>
class ndarray_iterator {
public:
  typedef T                                             ndarray_type;
  typedef typename ndarray_type::value_type               value_type;
  typedef typename ndarray_type::const_value_type   const_value_type;
  typedef typename ndarray_type::reference                 reference;
  typedef typename ndarray_type::const_reference     const_reference;
  typedef typename ndarray_type::pointer                     pointer;
  typedef typename ndarray_type::const_pointer         const_pointer;

protected:
  ndarray_type * _a;
  vector<int>    _index;
  vector<int>    _dimensions;

public:
  ndarray_iterator (ndarray_type * a, const vector<int> & index, const vector<int> & dimensions) {
    this->_a = a;
    this->_index = index;
    this->_dimensions = dimensions;
  }

  const vector<int> & index () const {
    return _index;
  }

  const_reference operator* () const {
    return (*_a).operator[](this->_index);
  }

  template<typename U = T, typename = enable_if_t<negation<is_const<U>>::value>>
  reference operator* () {
    return (*_a).operator[](this->_index);
  }

  pointer operator-> () {
    return &(this->operator*());
  }

  ndarray_iterator & operator++ () {
    for (int i = _dimensions.size() - 1; i >= 0; i--) {
      _index.at(_dimensions.at(i))++;
      if ((_index.at(_dimensions.at(i)) >= (*_a).shape().at(_dimensions.at(i))) && i != 0) {
        _index.at(_dimensions.at(i)) = 0;
      } else {
        break;
      }
    }
    return *this;
  }

  ndarray_iterator operator++ (int) {
    ndarray_iterator tmp(*this);
    ++(*this);
    return tmp;
  }

  bool operator== (const ndarray_iterator<T> & other) const {
    if (this->_dimensions.size() != other._dimensions.size() || this->_index.size() != other._index.size()) {
      return false;
    }
    for (int i = 0; i < this->_dimensions.size(); i++) {
      if (this->_dimensions.at(i) != other._dimensions.at(i)) {
        return false;
      }
    }
    for (int i = 0; i < this->_index.size(); i++) {
      if (this->_index.at(i) != other._index.at(i)) {
        return false;
      }
    }
    return true;
  }

  bool operator!= (const ndarray_iterator<T> & other) const {
    return !(this->operator==(other));
  }
};

template<typename T>
class ndarray {
public:
  typedef T                                                   value_type;
  typedef const value_type                              const_value_type;
  typedef value_type &                                         reference;
  typedef const_value_type &                             const_reference;
  typedef value_type *                                           pointer;
  typedef const_value_type *                               const_pointer;
  typedef ndarray_iterator<ndarray<value_type>>                 iterator;
  typedef ndarray_iterator<const ndarray<value_type>>     const_iterator;

protected:
  vector<value_type> _data;
  vector<int> _shape;
  vector<int> _offsets;
  value_type _min, _max;

public:
  ndarray () 
  { }
  
  // Some template metaprogramming...
  template<typename... U, typename = enable_if_t<conjunction<is_same<U, int>...>::value>>
  ndarray (const tuple<U...> & shape) {

  }

  ndarray (const vector<int> & shape) {
    int prod = 1;
    _shape = shape;
    _offsets.resize(_shape.size());
    for (int i = _shape.size() - 1; i >= 0; i--) {
      prod *= _shape.at(i);
      if (i == _shape.size() - 1) {
        _offsets.at(i) = 1;
      } else {
        _offsets.at(i) = _offsets.at(i + 1) * _shape.at(i + 1);
      }
    }
    _data.resize(prod);
  }

  ndarray (const ndarray<value_type> & other) {
    if (&other != this) {
      return;
    }
    this->_data = other._data;
    this->_shape = other._shape;
    this->_offsets = other._offsets;
    this->_min = other._min;
    this->_max = other._max;
  }

  const vector<int> & shape () const {
    return this->_shape;
  }

  const vector<int> & offsets () const {
    return this->_offsets;
  }

  const_reference operator[] (const vector<int> & coords) const {
    int index = 0;
    for (int i = 0; i < _offsets.size(); i++) {
      index += _offsets.at(i) * coords.at(i);
    }
    return _data.at(index);
  }

  reference operator[] (const vector<int> & coords) {
    int index = 0;
    for (int i = 0; i < _offsets.size(); i++) {
      index += _offsets.at(i) * coords.at(i);
    }
    return _data.at(index);
  }

  ndarray<value_type> & operator= (const ndarray<value_type> & other) {
    if (&other == this) {
      return *this;
    }
    this->_data = other._data;
    this->_shape = other._shape;
    this->_offsets = other._offsets;
    this->_min = other._min;
    this->_max = other._max;
    return *this;
  }

  ndarray<value_type> & operator= (ndarray<value_type> && other) {
    if (&other == this) {
      return *this;
    }
    cerr << "operator= Before for loop - shape: (";
    for (int i = 0; i < other.shape().size(); i++) {
      if (i < 3 || i > other.shape().size() - 4) {
        cerr << " " << other.shape().at(i);
      } else if (i == 4) {
        cerr << " ...";
      }
    }
    cerr << " )\n";
    this->_data = other._data;
    this->_shape = other._shape;
    this->_offsets = other._offsets;
    this->_min = other._min;
    this->_max = other._max;
    return *this;
  }

  const_reference max () const {
    return this->_max;
  }

  reference max () {
    return this->_max;
  }

  const_reference min () const {
    return this->_min;
  }

  reference min () {
    return this->_min;
  }

  template<typename U, typename = enable_if_t<negation<is_same<U, T>>::value>>
  ndarray<value_type> & operator= (const ndarray<U> & other) {
    cerr << "operator= Before for loop - shape: (";
    for (int i = 0; i < other.shape().size(); i++) {
      if (i < 3 || i > other.shape().size() - 4) {
        cerr << " " << other.shape().at(i);
      } else if (i == 4) {
        cerr << " ...";
      }
    }
    cerr << " )\n";
    long long int size = accumulate(::begin(other.shape()), ::end(other.shape()), 1, multiplies<int>());
    this->_data.resize(size);
    this->_shape = other.shape();
    typename ndarray<U>::const_iterator j = other.begin();   // ndarray<U>::iterator
    for (ndarray<T>::iterator i = begin(); i != end(); ++i, ++j) {
      *i = static_cast<T>(*j);
    }
    //this->_data = other._data;
    this->_offsets = other.offsets();
    this->_min = static_cast<T>(other.min());
    this->_max = static_cast<T>(other.max());
    return *this;
  }

  const_iterator begin () const {
    vector<int> dim(this->_shape.size());
    for (int i = 0; i < dim.size(); i++) {
      dim.at(i) = i;
    }
    return const_iterator(this, vector<int>(this->_shape.size(), 0), dim);
  }

  iterator begin () {
    vector<int> dim(this->_shape.size());
    for (int i = 0; i < dim.size(); i++) {
      dim.at(i) = i;
    }
    return iterator(this, vector<int>(this->_shape.size(), 0), dim);
  }

  const_iterator begin (const vector<int> & dimensions) const {
    return const_iterator(this, vector<int>(this->_shape.size(), 0), dimensions);
  }

  iterator begin (const vector<int> & dimensions) {
    return iterator(this, vector<int>(this->_shape.size(), 0), dimensions);
  }

  const_iterator end () const {
    vector<int> index(this->_shape.size(), 0);
    index.at(0) = _shape.at(0);
    vector<int> dim(this->_shape.size());
    for (int i = 0; i < dim.size(); i++) {
      dim.at(i) = i;
    }
    return const_iterator(this, index, dim);
  }

  iterator end () {
    vector<int> index(this->_shape.size(), 0);
    index.at(0) = _shape.at(0);
    vector<int> dim(this->_shape.size());
    for (int i = 0; i < dim.size(); i++) {
      dim.at(i) = i;
    }
    return iterator(this, index, dim);
  }

  const_iterator end (const vector<int> & dimensions) const {
    vector<int> index(this->_shape.size(), 0);
    index.at(0) = _shape.at(0);
    return const_iterator(this, index, dimensions);
  }

  iterator end (const vector<int> & dimensions) {
    vector<int> index(this->_shape.size(), 0);
    index.at(0) = _shape.at(0);
    return iterator(this, index, dimensions);
  }
};

unsigned char * read_until (const unordered_set<char> taboo, unsigned char * buffer, const int size, FILE * stream);

class image {
protected:
  string magic_number = "P5";
  //vector<vector<unsigned char>> data;
  ndarray<unsigned char> _data;
  short int _height, _width;
  short int _min, _max;
  long double _mean;

  void get_dimensions (FILE * fin) {
    const int BUFFER_SIZE = 1024;
    unsigned char buffer[BUFFER_SIZE];
    memset(buffer, 0, BUFFER_SIZE);
    string white_space(" \r\t\n");
    unordered_set<char> taboo(white_space.begin(), white_space.end());

    string magic_number = (char *)read_until(taboo, buffer, BUFFER_SIZE, fin);
    //fread(buffer, 3, 1, fin);
    cout << "Magic Number: " << magic_number << endl;
    memset(buffer, 0, BUFFER_SIZE);

    char * endpt;
    _width = strtol((char *)read_until(taboo, buffer, BUFFER_SIZE, fin), &endpt, 10);
    if (*endpt == '\0') {
      cout << "Width: " << _width << endl;
    }
    memset(buffer, 0, BUFFER_SIZE);

    _height = strtol((char *)read_until(taboo, buffer, BUFFER_SIZE, fin), &endpt, 10);
    if (*endpt == '\0') {
      cout << "Height: " << _height << endl;
    }
    memset(buffer, 0, BUFFER_SIZE);

    _max = strtol((char *)read_until(taboo, buffer, BUFFER_SIZE, fin), &endpt, 10);
    if (*endpt == '\0') {
      cout << "Max: " << _max << endl;
    }
  }

public:
  //void print();

  image () { }

  void print ();

  image (FILE * fin) {
    get_dimensions(fin);
    const short int READ_SIZE = _max < 256 ? 1 : 2;
    const int BUFFER_SIZE = _width * READ_SIZE;
    unsigned char buffer[BUFFER_SIZE];
    int pos = 0;
    //memset(buffer, 0, BUFFER_SIZE);

    //fread(buffer, READ_SIZE, width, fin);

    _min = numeric_limits<short int>::max();
    long long int sum = 0;

    //data.resize(_height);
    cerr << "Before ndarray\n";
    _data = ndarray<unsigned char>({_height, _width});
    cerr << "Before for loop - shape: (";
    for (int i = 0; i < _data.shape().size(); i++) {
      if (i < 3 || i > _data.shape().size() - 4) {
        cerr << " " << _data.shape().at(i);
      } else if (i == 4) {
        cerr << " ...";
      }
    }
    cerr << " )\n";
    cerr << "iterator {";
    int count = 0;
    for (ndarray<unsigned char>::iterator i = _data.begin(); i != _data.end(); ++i) {
      // data.at(i).resize(_width);
      // memset(buffer, 0, BUFFER_SIZE);
      if (pos * READ_SIZE == BUFFER_SIZE) {
        pos = 0;
      }
      if (pos == 0) {
        fread(buffer, READ_SIZE, _width, fin);
      }
      *i = buffer[pos * READ_SIZE];
      sum += *i;
      if (_min > *i) {
        _min = *i;
      }
      pos++;
      if (count < 10) {
        cout << " " << (int)*i;
        cout.flush();
      }
      count++;
    }
    cout << " count: " << count << " }\n";
    _mean = sum / ((double)(_height * _width));
    print();
  }

  image (const image & other) {
    this->magic_number = other.magic_number;
    this->_data = other._data;
    this->_height = other._height;
    this->_width = other._width;
    this->_min = other._min;
    this->_max = other._max;
    this->_mean = other._mean;
  }

  template<typename T>
  image (const ndarray<T> & matrix) {
    cerr << "before assign\n";
    _data = matrix;
    cerr << "after assign\n";
    _height = _data.shape().at(0);
    _width = _data.shape().at(1);
    _min = _data.max();
    _max = _data.min();
    //long long int sum = 0;
    //_data.resize(matrix.size());
    // _height = matrix.size();
    // _width = matrix.at(0).size();
    // for (int i = 0; i < matrix.size(); i++) {
    //   data.at(i).resize(matrix.at(i).size());
    //   for (int j = 0; j < matrix.at(i).size(); j++) {
    //     unsigned char item = (is_same<T, float>::value || is_same<T, double>::value) ? (unsigned char)(matrix.at(i).at(j) + 0.5) : (unsigned char)matrix.at(i).at(j);
    //     sum += item;
    //     if (_min > item) {
    //       _min = item;
    //     }
    //     if (_max < item) {
    //       _max = item;
    //     }
    //     data.at(i).at(j) = item;
    //   }
    // }
    // _mean = sum / ((double)_height * _width);
  }

  int output_to_PGM (const filesystem::path & p) {
    ofstream fout(p.string(), ios_base::out | ios_base::trunc | ios_base::binary);
    if (fout.fail()) {
      cerr << "Failed to open file at: " << p.string() << endl;
      return -1;
    }
    fout << magic_number << "\n" << to_string(_width) << " " << to_string(_height) << "\n" << to_string(_max) << "\n";
    // for (int i = 0; i < data.size(); i++) {
    //   for (int j = 0; j < data.at(i).size(); j++) {
    //     fout << data.at(i).at(j);
    //   }
    // }
    fout.close();
    return 0;
  }

  void output_to_JPG (filesystem::path path) {
    filesystem::path pgm_path = path.replace_extension(".PGM");
    path.replace_extension(".JPG");
    pid_t process_id;
    int status = output_to_PGM(pgm_path);
    if (status != 0) {
      return;
    }

    process_id = fork();
    if (process_id < 0) {
      perror("Fork error");
      exit(2);
    } else if (process_id == 0) {
      // Child process
      execl("/usr/bin/cjpeg", "/usr/bin/cjpeg", "-grayscale", "-outfile", path.c_str(), pgm_path.c_str(), (char *)NULL);
      perror("cjpeg not installed");
      exit(2);
    } else {
      // Parent process
      process_id = wait(&status);
      if (WIFEXITED(status) && WEXITSTATUS(status) != 0) {
        cerr << "cjpeg exited with unsuccessful status: " << WEXITSTATUS(status) << endl;
        exit(3);
      }
      status = remove(pgm_path.c_str());
      if (status != 0) {
        cerr << "Failed to remove: " << pgm_path.string() << endl;
      }
    }
  }

  pair<const short int &, const short int &> shape () const {
    return {this->_width, this->_height};
  }

  const short int & height () const {
    return this->_height;
  }

  const short int & width () const {
    return this->_width;
  }

  const short int & min () const {
    return this->_min;
  }

  const short int & max () const {
    return this->_max;
  }

  const long double & mean () const {
    return this->_mean;
  }

  image & operator= (const image & other) {
    if (&other == this) {
      return *this;
    }
    this->magic_number = other.magic_number;
    this->_data = other._data;
    this->_height = other._height;
    this->_width = other._width;
    this->_min = other._min;
    this->_max = other._max;
    this->_mean = other._mean;
    return *this;
  }

  image & operator= (image && other) {
    if (&other == this) {
      return *this;
    }
    this->magic_number = other.magic_number;
    this->_data = other._data;
    this->_height = other._height;
    this->_width = other._width;
    this->_min = other._min;
    this->_max = other._max;
    this->_mean = other._mean;
    return *this;
  }

  unsigned char & operator[] (const pair<int, int> & index) {
    return _data[{index.first, index.second}]; // data.at(index.first).at(index.second);
  }

  const unsigned char & operator[] (const pair<int, int> & index) const {
    return _data[{index.first, index.second}]; // data.at(index.first).at(index.second);
  }
};

void image::print () {
  cout << ">Print-----------------------------------------------------------------------------------\n";
  cout << "Min: " << _min << " Max: " << _max << " Mean: " << setprecision(14) << _mean << "\nData: \n[";
  for (int i = 0; i < _height; i++) {
    if (i < 3 || i > _height - 4) {
      if (i != 0) {
        cout << "\n ";
      }
      cout << "[";
    } else if (i == 4) {
      cout << "\n ...";
    }
    for (int j = 0; j < _width; j++) {
      if ((j < 3 || j > _width - 4) && (i < 3 || i > _height - 4)) {
        cout << " " << setw(3) << (int)_data[{i, j}];
      } else if (j == 3 && (i < 3 || i > _height - 4)) {
        cout << " ...";
      }
    }
    if (i < 3 || i > _height - 4) {
      cout << "]";
    }
  }
  cout << "]\n";
  cout << "-----------------------------------------------------------------------------------------\n";
}

class colony {
protected:
  filesystem::path img_path;
  image img;
  double alpha, beta;
  int i_max;
  ndarray<double> intensities;

  short int get_intensity_val (const int row, const int col) {
    return max<short int> (
      max<short int>(
        (row - 1 >= 0 && col - 1 >= 0 && row + 1 < img.height() && col + 1 < img.width()) ? 
        abs(((short int)img[{row - 1, col - 1}]) - ((short int)img[{row + 1, col + 1}])) : 0
        ,
        (row - 1 >= 0 && col - 1 >= 0 && row + 1 < img.height() && col + 1 < img.width()) ? 
        abs(((short int)img[{row - 1, col + 1}]) - ((short int)img[{row + 1, col - 1}])) : 0
      ),
      max<short int>(
        (row - 1 >= 0 && row + 1 < img.height()) ? 
        abs(((short int)img[{row - 1, col}]) - ((short int)img[{row + 1, col}])) : 0
        ,
        (col - 1 >= 0 && col + 1 < img.width()) ? 
        abs(((short int)img[{row, col - 1}]) - ((short int)img[{row, col + 1}])) : 0
      )
    );
  }

  void setup_intensities (const bool normalize = true) {
    short int max = numeric_limits<short int>::min();
    this->intensities = ndarray<double>({img.height(), img.width()});
    for (ndarray<double>::iterator i = this->intensities.begin(); i != this->intensities.end(); ++i) {
      int row = i.index().at(0);
      int col = i.index().at(1);
      short int val = get_intensity_val(row, col);
      *i = val;
      if (val > max) {
        max = val;
      }
    }
    if (normalize == true) {
      for (ndarray<double>::iterator i = this->intensities.begin(); i != this->intensities.end(); ++i) {
        *i /= max;
      }
    }
    // for (int i = 0; i < this->intensities.size(); i++) {
    //   this->intensities.at(i).resize(img.width());
    //   for (int j = 0; j < this->intensities.at(i).size(); j++) {
    //     this->intensities.at(i).at(j) = get_intensity_val(i, j);
    //     if (this->intensities.at(i).at(j) > max) {
    //       max = this->intensities.at(i).at(j);
    //     }
    //   }
    // }
    // if (normalize == true) {
    //   for (int i = 0; i < this->intensities.size(); i++) {
    //     for (int j = 0; j < this->intensities.at(i).size(); j++) {
    //       this->intensities.at(i).at(j) /= max;
    //     }
    //   }
    // }
  }

public:
  void output_intensities (const bool binary_print = true);

  colony (const filesystem::path & img_path, const image & img, const int ant_count = -1, 
          const double pheromone_evaporation_constant = 0.1, const int pheromone_memory_constant = 20,
          const int ant_memory_constant = 20, const double minimum_pheromone_constant = 0.0001, 
          const double intensity_threshold_value = -1.0, const double alpha = 1.0, const double beta = 1.0) {
    this->alpha = alpha;
    this->beta = beta;
    this->img_path = img_path;
    this->img = img;
    this->i_max = img.max();
    setup_intensities();
    output_intensities();
  }
};

void colony::output_intensities (const bool binary_print) {
  ndarray<double> intensities_copy = intensities;
  //intensities_copy.print();
  for (ndarray<double>::iterator i = intensities_copy.begin(); i != intensities_copy.end(); ++i) {
    *i = (int)((((255 - 0) * (*i - intensities_copy.min())) / (intensities_copy.max() - intensities_copy.min())) + 0 + 0.5);
  }
  image intensities_image(intensities_copy);
  filesystem::path dir = img_path.parent_path();
  dir.append("Intensities");
  dir /= img_path.filename();
  cout << "Established intensity dir: " << dir.string() << endl;
  intensities_image.print();
  intensities_image.output_to_JPG(dir);
}

unsigned char * read_until (const unordered_set<char> taboo, unsigned char * buffer, const int size, FILE * stream) {
  for (int i = 0; i < size; i++) {
    char c = fgetc(stream);
    if (taboo.find(c) != taboo.end()) {
      cout << endl;
      return buffer;
    } else {
      cout << "Buffer receiving: '" << c << "' ";
      buffer[i] = c;
    }
  }
  cout << endl;
  return buffer;
}

image read_PGM (const filesystem::path & path) {
  cout.flush();
  cerr.flush();

  int p[2], status;
  status = pipe(p);
  pid_t process_id;

  if (status < 0) {
    perror("Pipe error");
    exit(2);
  }

  process_id = fork();
  if (process_id < 0) {
    perror("Fork error");
    exit(2);
  } else if (process_id == 0) {
    // Child process
    close(p[0]);  // Child will not read
    close(1);     // Close stdout
    dup(p[1]);    // Duplicate write end as stdout
    execl("/usr/bin/djpeg", "/usr/bin/djpeg", "-pnm", "-grayscale", path.c_str(), (char *)NULL);
    perror("djpeg not installed");
    exit(2);
  } else {
    // Parent process
    close(p[1]);  // Parent will not write
    FILE * fin = fdopen(p[0], "rb");
    image img = image(fin);
    fclose(fin);
    process_id = wait(&status);
    if (WIFEXITED(status) && WEXITSTATUS(status) != 0) {
      cerr << "djpeg exited with unsuccessful status: " << WEXITSTATUS(status) << endl;
      exit(3);
    }
    return img;
  }
}

int main (int argc, char * argv[]) {
  string s;
  cout << "Args: [";
  for (int i = 0; i < argc; i++) {
    cout << "'" << argv[i] << "', ";
  }
  cout << "\b\b]\n";
  if (argc != 2) {
    cerr << "Usage: " << argv[0] << ": `"  << filesystem::path(argv[0]).filename().string() << " <image directory>'\n";
    exit(1);
  }
  cout << "Directory [" << argv[1] << "]: [";
  for (const filesystem::directory_entry & entry : filesystem::directory_iterator(argv[1])) {
    cout << "'" << entry.path().filename().string() << "', ";
  }
  cout << "\b\b]\n";

  for (const filesystem::directory_entry & entry : filesystem::directory_iterator(argv[1])) {
    string item_name = entry.path().filename().string();
    if (item_name == "Intensities" || item_name == "Iterations" || item_name == "Intensities-test") {
      continue;
    }
    filesystem::path dir(argv[1]);
    filesystem::path p = dir / entry.path();
    cout << "Path: " << p.string() << endl;
    image img = read_PGM(p);
    colony c(p, img);
    break;
  }

  return 0;
}
