mod extract_features;
mod kmeans;

const MAX_ITERATIONS: i32 = 1000;

fn normalize_features(features: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    features
        .iter()
        .map(|feature| {
            let min = feature.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = feature.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            if (max - min).abs() < std::f64::EPSILON {
                // Avoid division by zero if all values are the same
                vec![0.0; feature.len()]
            } else {
                feature.iter().map(|v| (v - min) / (max - min)).collect()
            }
        })
        .collect()
}

fn main() {
    let outlines = extract_features::read_outlines("/Users/sam/dev/kmeans/src/9_outlines.txt").expect("Failed to read outlines");
    let mut centroids = extract_features::calculate_centroids(&outlines);
    //println!("{:?}", features[0][0].0);

    let hull = extract_features::convex_hull(outlines.clone());
    let mut areas = Vec::new();
    for cell in &outlines {
        let area = extract_features::convex_area(cell);
        areas.push(area);
    }
    //println!("{:?}", hull[0]);

    for cell in &outlines {
        //println!("{:?} {:?}", extract_features::convex_area(cell), extract_features::convex_perimeter(cell));
    }

    let rgb_matrix = kmeans::load_image_as_matrix("/Users/sam/dev/kmeans/src/9.png");
    let segmentation_rgb = extract_features::load_segmentations_as_matrix("/Users/sam/dev/kmeans/src/9.png", "/Users/sam/dev/kmeans/src/9_outlines.txt");
    let channel_means = extract_features::channel_mean(&segmentation_rgb);
    //println!("{:?}", extract_features::channel_mean(&segmentation_rgb));

    let voronoi = extract_features::voronoi_areas(&centroids, 512, 512);
    for area in &voronoi {
        println!("{:?}", area);
    }

    let centroid_x: Vec<f64> = centroids.iter().map(|c| c.0 as f64).collect();
    let centroid_y: Vec<f64> = centroids.iter().map(|c| c.1 as f64).collect();
    let channel_mean_red: Vec<f64> = channel_means.iter().map(|c| c.0 as f64).collect();
    let channel_mean_green: Vec<f64> = channel_means.iter().map(|c| c.1 as f64).collect();
    let channel_mean_blue: Vec<f64> = channel_means.iter().map(|c| c.2 as f64).collect();
    let voronoi_f64: Vec<f64> = voronoi.iter().map(|a| *a as f64).collect();
    let features: Vec<Vec<f64>> = vec![centroid_x, centroid_y, areas, channel_mean_red, channel_mean_green, channel_mean_blue, voronoi_f64];
    let features = normalize_features(&features);

    let mut iterations = 0;
    let k = 10;
    let mut centroids = centroids.iter().map(|x| (x.0 as usize, x.1 as usize)).collect();
    let mut old_centroids: Vec<(usize, usize)> = Vec::new();
    let mut final_labels_matrix: Vec<Vec<usize>> = vec![vec![0; features[0].len()]; features.len()];
    while !kmeans::should_stop(&old_centroids, &centroids, iterations) {
        iterations += 1;

        let (labels_from_this_iteration, by_labels) = kmeans::get_labels(&features, &centroids);
        final_labels_matrix = labels_from_this_iteration;
        //println!("{:?}", labels[0][0]);

        old_centroids = centroids;
        centroids = kmeans::get_centroids(&by_labels);
    }

    println!("K-means clustering finished after {} iterations.", iterations);
    println!("Final centroids: {:?}", centroids);

    // Now, call the visualization functions:
    let colors = kmeans::label_colors(k); // Generate `k` distinct colors for the clusters.
    
    kmeans::save_clustered_image(
        &final_labels_matrix,       // The final cluster assignments for each pixel.
        &colors,                    // The colors for each cluster.
        features[0].len() as u32,               // Image width.
        features.len() as u32,               // Image height.
        "/Users/sam/dev/kmeans/src/clustered.png" // Your desired output path.
    );

}

// DECODE: DEep Cell Observation & Discovery Engine