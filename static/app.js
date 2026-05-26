document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const vehicleCountEl = document.getElementById('vehicle-count');
    const warningBanner = document.getElementById('congestion-warning');
    const currentStateEl = document.getElementById('current-state');
    const timeLeftEl = document.getElementById('time-left');
    
    // Traffic Light Elements
    const lightRed = document.getElementById('light-red');
    const lightYellow = document.getElementById('light-yellow');
    const lightGreen = document.getElementById('light-green');
    
    // Helper to update traffic light UI
    function updateTrafficLight(state) {
        // Reset all lights
        lightRed.classList.remove('active');
        lightYellow.classList.remove('active');
        lightGreen.classList.remove('active');
        
        currentStateEl.textContent = state;
        
        // Activate current light
        if (state === 'RED') {
            lightRed.classList.add('active');
            currentStateEl.style.color = 'var(--red-glow)';
        } else if (state === 'YELLOW') {
            lightYellow.classList.add('active');
            currentStateEl.style.color = 'var(--yellow-glow)';
        } else if (state === 'GREEN') {
            lightGreen.classList.add('active');
            currentStateEl.style.color = 'var(--green-glow)';
        }
    }
    
    // Initialize Chart.js
    const ctx = document.getElementById('densityChart').getContext('2d');
    
    // Create beautiful neon cyan gradient
    const gradient = ctx.createLinearGradient(0, 0, 0, 180);
    gradient.addColorStop(0, 'rgba(6, 182, 212, 0.4)');   // Neon Cyan at top
    gradient.addColorStop(0.5, 'rgba(59, 130, 246, 0.15)'); // Royal Blue in middle
    gradient.addColorStop(1, 'rgba(30, 41, 59, 0)');        // Fade out
    
    const densityChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: Array(20).fill(''), // X-axis labels
            datasets: [
                {
                    label: 'Live ROI Count',
                    data: Array(20).fill(0),
                    borderColor: '#06b6d4', // Cyan border
                    backgroundColor: gradient,
                    borderWidth: 3,
                    pointRadius: 2,
                    pointBackgroundColor: '#06b6d4',
                    pointHoverRadius: 5,
                    tension: 0.4,
                    fill: true
                },
                {
                    label: 'Warning Limit',
                    data: Array(20).fill(6), // Default to 6
                    borderColor: '#ef4444',  // Red border
                    borderWidth: 1.5,
                    borderDash: [5, 5],
                    pointRadius: 0,
                    fill: false,
                    tension: 0
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    suggestedMax: 10,
                    grid: {
                        color: 'rgba(255, 255, 255, 0.05)'
                    },
                    ticks: {
                        color: '#94a3b8',
                        font: {
                            family: 'Inter',
                            size: 11
                        }
                    }
                },
                x: {
                    grid: {
                        display: false
                    },
                    ticks: {
                        display: false
                    }
                }
            },
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    align: 'end',
                    labels: {
                        color: '#94a3b8',
                        font: {
                            family: 'Inter',
                            size: 11,
                            weight: '500'
                        },
                        boxWidth: 10,
                        usePointStyle: true,
                        pointStyle: 'rectRounded'
                    }
                },
                tooltip: {
                    backgroundColor: '#1e293b',
                    titleColor: '#f8fafc',
                    bodyColor: '#cbd5e1',
                    borderColor: 'rgba(255, 255, 255, 0.1)',
                    borderWidth: 1,
                    padding: 8,
                    cornerRadius: 8,
                    displayColors: true,
                    boxWidth: 8,
                    boxHeight: 8,
                    usePointStyle: true
                }
            },
            interaction: {
                intersect: false,
                mode: 'index'
            },
            animation: {
                duration: 150
            }
        }
    });

    // Fetch metrics from backend periodically
    async function fetchMetrics() {
        try {
            const response = await fetch('/api/metrics');
            const data = await response.json();
            
            // Update Vehicle Count
            vehicleCountEl.textContent = data.vehicles_in_roi;
            
            // Update Warning Banner
            if (data.is_congested) {
                warningBanner.classList.remove('hidden');
            } else {
                warningBanner.classList.add('hidden');
            }
            
            // Update Traffic Light
            updateTrafficLight(data.state);
            
            // Update Time Left
            timeLeftEl.textContent = data.time_left.toFixed(1) + 's';
            
            // Update Vehicle Breakdown
            if (data.live_counts && data.total_counts) {
                document.getElementById('count-cars').innerHTML = `${data.live_counts.car} <span class="total-badge">Total: ${data.total_counts.car}</span>`;
                document.getElementById('count-bikes').innerHTML = `${data.live_counts.motorcycle} <span class="total-badge">Total: ${data.total_counts.motorcycle}</span>`;
                document.getElementById('count-buses').innerHTML = `${data.live_counts.bus} <span class="total-badge">Total: ${data.total_counts.bus}</span>`;
                document.getElementById('count-trucks').innerHTML = `${data.live_counts.truck} <span class="total-badge">Total: ${data.total_counts.truck}</span>`;
            }
            
            // Update Total Tracked Vehicles Count
            if (data.total_vehicles !== undefined) {
                document.getElementById('total-vehicles').textContent = data.total_vehicles;
            }
            
            // Update Chart
            const currentData = densityChart.data.datasets[0].data;
            currentData.shift(); // Remove oldest
            currentData.push(data.vehicles_in_roi); // Add newest
            
            // Update dynamic warning limit from backend threshold
            if (data.congestion_threshold !== undefined) {
                densityChart.data.datasets[1].data = Array(20).fill(data.congestion_threshold);
            }
            
            densityChart.update();
            
        } catch (error) {
            console.error("Error fetching metrics:", error);
        }
    }

    // Poll every 500ms
    setInterval(fetchMetrics, 500);
});
