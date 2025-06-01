//! cargo-deps: csv = "1.3.1"

/*
Extract features into a 2D vector of N cells each with roughly 20 features 
*/

use crate::kmeans::{self, load_image_as_matrix, Point};

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::error::Error;
use std::path::absolute;

pub fn read_outlines(filename: &str) -> Result<Vec<Vec<(i32, i32)>>, Box<dyn Error>> {
    let file = File::open(filename)?;
    let reader = BufReader::new(file);
    let mut data = Vec::new();
    
    for line in reader.lines() {
        let line = line?;
        // Split by comma, trim and parse integers.
        let numbers: Vec<i32> = line.split(',')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .map(|s| s.parse::<i32>())
            .collect::<Result<Vec<_>, _>>()?;
        
        if numbers.len() % 2 != 0 {
            return Err("Odd number of coordinates in line".into());
        }
        
        let pairs: Vec<(i32, i32)> = numbers.chunks(2)
            .map(|chunk| (chunk[0], chunk[1]))
            .collect();
        data.push(pairs);
    }
    Ok(data)
}

pub fn cross(a: (i32, i32), b: (i32, i32), c: (i32, i32)) -> i32 {
    (b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0)
}

pub fn convex_hull(mut points: Vec<Vec<(i32, i32)>>) -> Vec<Vec<(i32, i32)>> {
    let mut hulls: Vec<Vec<(i32, i32)>> = Vec::new();

    for cell in &mut points {
        cell.sort();
        // Build lower hull
        let mut lower: Vec<(i32, i32)> = vec![cell[0], cell[1]];
        for i in 2..cell.len() {
                if cross(cell[i - 2], cell[i - 1], cell[i]) <= 0 {
                    lower.pop();
                }

                lower.push(cell[i]);
        }

        // Build upper hull
        let mut upper: Vec<(i32, i32)> = vec![cell[cell.len() - 1], cell[cell.len() - 2]];
        for i in (0..cell.len() - 2).rev() {
            if  cross(upper[upper.len() - 2], upper[upper.len() - 1], cell[i]) <= 0 {
                upper.pop();
            }
            upper.push(cell[i]);
        }

        lower.pop();
        upper.pop();
        let mut hull = lower;
        hull.extend(upper);
        hulls.push(hull);
    }
    

    hulls
}

pub fn convex_area(points: &[(i32, i32)]) -> f64 {
    let mut area = 0.0;

    for i in 0..points.len() {
        if i == points.len() - 1 {
            area += (points[i].0 * points[0].1 - points[0].0 * points[i].1) as f64;
        } else {
            area += (points[i].0 * points[i+1].1 - points[i+1].0 * points[i].1) as f64;
        }
    }

    0.5 * area.abs()
}

pub fn convex_perimeter(points: &[(i32, i32)]) -> f64 {
    let mut perimeter = 0.0;
    let n  = points.len();

    for i in 0..n {
        let dx = (points[(i + 1) % n].0 - points[i].0) as f64;
        let dy = (points[(i + 1) % n].1 - points[i].1) as f64;
        perimeter += (dx * dx + dy * dy).sqrt();
    }

    perimeter
}

pub fn load_segmentations_as_matrix(image_path: &str, outline_path: &str) -> Vec<Vec<Point>> {
    let rgb_matrix: Vec<Vec<Point>> = load_image_as_matrix(image_path);
    let segmentation_matrix = read_outlines(outline_path).expect("Failed to load segmentations");
    let mut segmentation_rgb: Vec<Vec<Point>> = vec![vec![Point(0, 0, 0); segmentation_matrix[0].len()]; segmentation_matrix.len()];

    for (i, row) in segmentation_matrix.iter().enumerate() {
        for pixel in row {
            segmentation_rgb[i].push(rgb_matrix[pixel.1 as usize][pixel.0 as usize].clone());
        }
    }

    segmentation_rgb
}

pub fn channel_mean(channels: &Vec<Vec<kmeans::Point>>) -> Vec<(f64, f64, f64)> {
    let mut means: Vec<(f64, f64, f64)> = vec![(0.0, 0.0, 0.0); channels.len()];
    for (i, row) in channels.iter().enumerate() {
        for pixel in row {
            means[i].0 += pixel.0 as f64;
            means[i].1 += pixel.1 as f64;
            means[i].2 += pixel.2 as f64;
        }

        means[i] = (means[i].0 / row.len() as f64, means[i].1 / row.len() as f64, means[i].2 / row.len() as f64);
    }

    means
}

/*pub fn main() {
    let features = read_outlines("/Users/sam/dev/kmeans/src/7_outlines.txt").expect("Failed to read outlines");
    //println!("{:?}", features[0][0].0);

    let hull = convex_hull(features.clone());
    //println!("{:?}", hull[0]);

    for cell in &features {
        //println!("{:?}", convex_area(cell));
    }

    println!("{:?}", channel_mean(&features));
}*/