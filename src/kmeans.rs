use image::{ImageReader, RgbImage, Rgb};
use rand::Rng;

const MAX_ITERATIONS: i32 = 1000;

#[derive(Debug, Clone)]
pub struct Point(pub u8, pub u8, pub u8);

impl Point {
    pub fn sum(&self) -> i32 {
        (self.0 as i32) + (self.1 as i32) + (self.2 as i32)
    }
}

impl From<[u8; 3]> for Point {
    fn from(arr: [u8; 3]) -> Self {
        Point(arr[0], arr[1], arr[2])
    }
}
struct Image {
    x: usize,
    y: usize,
    img: Vec<Vec<Point>>,
}

pub fn load_image_as_matrix(path: &str) -> Vec<Vec<Point>> {
    let img = ImageReader::open(path)
        .expect("Failed to open image")
        .decode()
        .expect("Failed to decode image")
        .to_rgb8();

    let (width, height) = img.dimensions();
    let mut matrix = vec![vec![Point(0, 0, 0); width as usize]; height as usize];

    //let pix: &image::Rgb<u8>;

    for (x, y, pixel) in img.enumerate_pixels() {
        matrix[y as usize][x as usize] = Point::from(pixel.0);
    }

    matrix
}


fn euclidean_distance(p: &Vec<f64>, q: &Vec<f64>) -> f64 {
    p.iter()
        .zip(q.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt()
}

pub fn random_point(v: &Vec<Vec<f64>>) -> (usize, usize) {
    let mut rng = rand::thread_rng();
    let y = rng.gen_range(0..v.len());
    let x = rng.gen_range(0..v[0].len());
    (x, y)
}

pub fn initialize_centroids(v: &Vec<Vec<f64>>) -> Vec<(usize, usize)> {
    let mut centroids = Vec::new();

    for i in 0..v.len() {
        centroids.push(random_point(v));
    }

    centroids
}

pub fn get_labels(img: &Vec<Vec<f64>>, centroids: &[(usize, usize)]) -> (Vec<Vec<usize>>, Vec<Vec<(usize, usize)>>) {
    let mut labels = vec![vec![0; img[0].len()]; img.len()];
    let mut by_label: Vec<Vec<(usize, usize)>> = vec![Vec::new(); centroids.len()];

    for (y, row) in img.iter().enumerate() {
        for (x, pixel) in row.iter().enumerate() {
            let mut min_dist = usize::MAX;
            let mut label = 0;
            for (i, &(cx, cy)) in centroids.iter().enumerate() {
                let dy = y as isize - cy as isize;
                let dx = x as isize - cx as isize;
                let dist = ((dy * dy + dx * dx) as usize).isqrt();
                if dist < min_dist {
                    min_dist = dist;
                    label = i;
                }
            }

            by_label[label].push((x, y));
            labels[y][x] = label;
        }
    }

    (labels, by_label)
}

impl Image {
    pub fn filter_image(&mut self) {
        for row in &mut self.img {
            for pixel in row {
                if pixel.sum() < 80 {
                    *pixel = Point(0, 0, 0);
                }
            }
        }
    }
}

pub fn get_centroids(labels: &Vec<Vec<(usize, usize)>>) -> Vec<(usize, usize)> {
    let mut centroids: Vec<(usize, usize)> = Vec::new();

    for row in labels.iter() {
        let mut centroid: (usize, usize)  = (0, 0);

        for point in row.iter() {
            centroid.0 += point.0;
            centroid.1 += point.1;
            }

        centroid.0 /= if row.len() != 0 {row.len()} else {1};
        centroid.1 /= if row.len() != 0 {row.len()} else {1};
        centroids.push(centroid);
    }

    centroids
}

pub fn should_stop(old_centroids: &Vec<(usize, usize)>, centroids: &Vec<(usize, usize)>, iterations: i32) -> bool {
    if iterations > MAX_ITERATIONS { return true };
    old_centroids == centroids
}

pub fn label_colors(num_colors: usize) -> Vec<Rgb<u8>> {
    let mut colors = Vec::new();
    let mut rng = rand::rng(); // Uses the same rand::rng() style as your existing code

    // Predefined distinct colors for the first few clusters
    let predefined_colors = [
        Rgb([255, 0, 0]),   // Red
        Rgb([0, 255, 0]),   // Green
        Rgb([0, 0, 255]),   // Blue
        Rgb([255, 255, 0]), // Yellow
        Rgb([0, 255, 255]), // Cyan
        Rgb([255, 0, 255]), // Magenta
        Rgb([192, 192, 192]),// Silver
        Rgb([128, 0, 0]),   // Maroon
        Rgb([128, 128, 0]), // Olive
        Rgb([0, 128, 0]),   // Dark Green
        Rgb([128, 0, 128]), // Purple
        Rgb([0, 128, 128]), // Teal
        Rgb([255, 165, 0]), // Orange
        Rgb([255, 192, 203]),// Pink
        Rgb([75, 0, 130]),  // Indigo
    ];

    for i in 0..num_colors {
        if i < predefined_colors.len() {
            colors.push(predefined_colors[i]);
        } else {
            // Generate random colors for additional clusters
            // This assumes rng.gen::<u8>() is compatible with your rand version.
            // It's a common method on the Rng trait.
            colors.push(Rgb([
                rng.random::<u8>(),
                rng.random::<u8>(),
                rng.random::<u8>(),
            ]));
        }
    }
    colors
}

// Saves the clustered image.
// `final_labels_matrix[y][x]` contains the cluster index for pixel (x,y).
// `img_width` and `img_height` are the dimensions of the image.
pub fn save_clustered_image(
    final_labels_matrix: &Vec<Vec<usize>>,
    cluster_colors: &Vec<Rgb<u8>>,
    img_width: u32,
    img_height: u32,
    output_path: &str,
) {
    let mut output_image = RgbImage::new(img_width, img_height);

    for y_idx in 0..img_height {
        for x_idx in 0..img_width {
            // Check bounds for labels_matrix access
            if (y_idx as usize) < final_labels_matrix.len() && (x_idx as usize) < final_labels_matrix[y_idx as usize].len() {
                let label_idx = final_labels_matrix[y_idx as usize][x_idx as usize];
                
                if label_idx < cluster_colors.len() {
                    let color = cluster_colors[label_idx];
                    output_image.put_pixel(x_idx, y_idx, color);
                } else {
                    // Fallback for label_idx out of bounds of cluster_colors
                    // This might happen if k is larger than predefined colors and random generation isn't enough,
                    // or if a label index is unexpectedly high.
                    output_image.put_pixel(x_idx, y_idx, Rgb([30, 30, 30])); // Dark gray fallback
                    // Only print this warning once to avoid spamming
                    if y_idx == 0 && x_idx == 0 && !cluster_colors.is_empty() { // Example: print for first pixel only
                         eprintln!(
                            "Warning: A label index ({}) is out of bounds for cluster_colors array (len {}). Using fallback color. (This warning is shown once)",
                            label_idx, cluster_colors.len()
                        );
                    }
                }
            } else {
                 // This case means img_width/img_height mismatch with final_labels_matrix dimensions
                 output_image.put_pixel(x_idx, y_idx, Rgb([0,0,0])); // Black for out-of-bounds pixel access
                 if y_idx == 0 && x_idx == 0 { // Print warning once
                    eprintln!(
                        "Warning: Pixel coordinates ({}, {}) are out of bounds for labels_matrix. (This warning is shown once)",
                        x_idx, y_idx
                    );
                 }
            }
        }
    }

    match output_image.save(output_path) {
        Ok(_) => println!("Clustered image saved as {}", output_path),
        Err(e) => eprintln!("Error: Failed to save clustered image to {}: {}", output_path, e),
    }
}

/*pub fn main() {
    let rgb_matrix = load_image_as_matrix("/Users/sam/dev/kmeans/src/9.png");
    let mut img = Image {
        x: rgb_matrix[0].len(),
        y: rgb_matrix.len(),
        img: rgb_matrix.clone()
    };
    println!("Loaded image with {} rows and {} columns", rgb_matrix.len(), rgb_matrix[0].len());
    println!("{:?}", rgb_matrix[0][0]);

    let k = 100;
    let mut centroids = initialize_centroids(k, &img);
    println!("{centroids:?}");

    let mut iterations = 0;
    let mut old_centroids: Vec<(usize, usize)> = Vec::new();

    img.filter_image();
    println!("{:?}", img.img[0][0]);

    let mut final_labels_matrix: Vec<Vec<usize>> = vec![vec![0; img.x]; img.y];
    while !should_stop(&old_centroids, &centroids, iterations) {
        iterations += 1;

        let (labels_from_this_iteration, by_labels) = get_labels(&img, &centroids);
        final_labels_matrix = labels_from_this_iteration;
        //println!("{:?}", labels[0][0]);

        old_centroids = centroids;
        centroids = get_centroids(&by_labels);
    }

    println!("K-means clustering finished after {} iterations.", iterations);
    println!("Final centroids: {:?}", centroids);

    // Now, call the visualization functions:
    let colors = label_colors(k); // Generate `k` distinct colors for the clusters.
    
    save_clustered_image(
        &final_labels_matrix,       // The final cluster assignments for each pixel.
        &colors,                    // The colors for each cluster.
        img.x as u32,               // Image width.
        img.y as u32,               // Image height.
        "/Users/sam/dev/kmeans/src/clustered.png" // Your desired output path.
    );
}*/
