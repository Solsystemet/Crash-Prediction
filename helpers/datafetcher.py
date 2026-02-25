import requests
import json

def data_fetcher(filename: str):
    """
    Fetch Chicago traffic crash data from Socrata API using POST request.
    Saves the CSV response to the specified filename.
    """
    url = "https://data.cityofchicago.org/api/v3/views/85ca-t3if/export.csv?cacheBust=1771951320&accessType=DOWNLOAD"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:147.0) Gecko/20100101 Firefox/147.0",
        "Accept": "text/csv",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br, zstd",
        "Referer": "https://data.cityofchicago.org/Transportation/Traffic-Crashes-Crashes/85ca-t3if/explore/query/SELECT%0A%20%20%60crash_record_id%60%2C%0A%20%20%60crash_date_est_i%60%2C%0A%20%20%60crash_date%60%2C%0A%20%20%60posted_speed_limit%60%2C%0A%20%20%60traffic_control_device%60%2C%0A%20%20%60device_condition%60%2C%0A%20%20%60weather_condition%60%2C%0A%20%20%60lighting_condition%60%2C%0A%20%20%60first_crash_type%60%2C%0A%20%20%60trafficway_type%60%2C%0A%20%20%60lane_cnt%60%2C%0A%20%20%60alignment%60%2C%0A%20%20%60roadway_surface_cond%60%2C%0A%20%20%60road_defect%60%2C%0A%20%20%60report_type%60%2C%0A%20%20%60crash_type%60%2C%0A%20%20%60intersection_related_i%60%2C%0A%20%20%60private_property_i%60%2C%0A%20%20%60hit_and_run_i%60%2C%0A%20%20%60damage%60%2C%0A%20%20%60date_police_notified%60%2C%0A%20%20%60prim_contributory_cause%60%2C%0A%20%20%60sec_contributory_cause%60%2C%0A%20%20%60street_no%60%2C%0A%20%20%60street_direction%60%2C%0A%20%20%60street_name%60%2C%0A%20%20%60beat_of_occurrence%60%2C%0A%20%20%60photos_taken_i%60%2C%0A%20%20%60statements_taken_i%60%2C%0A%20%20%60dooring_i%60%2C%0A%20%20%60work_zone_i%60%2C%0A%20%20%60work_zone_type%60%2C%0A%20%20%60workers_present_i%60%2C%0A%20%20%60num_units%60%2C%0A%20%20%60most_severe_injury%60%2C%0A%20%20%60injuries_total%60%2C%0A%20%20%60injuries_fatal%60%2C%0A%20%20%60injuries_incapacitating%60%2C%0A%20%20%60injuries_non_incapacitating%60%2C%0A%20%20%60injuries_reported_not_evident%60%2C%0A%20%20%60injuries_no_indication%60%2C%0A%20%20%60injuries_unknown%60%2C%0A%20%20%60crash_hour%60%2C%0A%20%20%60crash_day_of_week%60%2C%0A%20%20%60crash_month%60%2C%0A%20%20%60latitude%60%2C%0A%20%20%60longitude%60%2C%0A%20%20%60location%60%0AWHERE%0A%20%20%60crash_date%60%0A%20%20%20%20BETWEEN%20%222024-01-01T11%3A07%3A33%22%20%3A%3A%20floating_timestamp%0A%20%20%20%20AND%20%222024-01-30T11%3A07%3A33%22%20%3A%3A%20floating_timestamp%0AORDER%20BY%20%60crash_date%60%20DESC%20NULL%20FIRST%2C%20%60crash_record_id%60%20ASC%20NULL%20LAST/page/filter",
        "Content-Type": "application/json",
        "X-CSRF-Token": "UJYT0IG17E5dIJRGRcstFuUrxTR62o7pITdYYW8r5vjn2sWpSipAmGuDqHctZcFbAtG75rh3AYZsqbbUceblTA",
        "X-App-Token": "U29jcmF0YS0td2VraWNrYXNz0",
        "Origin": "https://data.cityofchicago.org",
        "Sec-GPC": "1",
        "Connection": "keep-alive",
        "Cookie": "_frontend_session=SUJYM2lRbEJFOHB2YTBqSWt3d2FKZ2VDckxWbStZTmp1MXluYWNVQkpSVE0vZ1BsUzhBMnJRUnQvNFZOdFpwUThoM0tlVUE2aHRlQ1FZaW0vaDZnVVZxVVVsV3F5eDlKVWM1Qjl1U01WSld0Z1lvZnJiZUN1Z01yMmlucC9iZjV4dEd6Wm11UTJ0N3JROEM3Vm15ODVkYXIrNC9IWkJHN2xyN0hLMjRJWjdKcFZRcUx4ZUcxZkdDaWJ3MDJjZi9FWjk3WGFrUlB0RTU3cDQwVFNmc0RONnNXY1BYM3FNTklzZ3luRXVPTXRGbnBxNy9UVUdHNFRpUXhjYVhvZ1crSHhHc21iM3BlRkt3bkVDNnI0SFVBWjRDanlnZXJpRkI5eG5CeFpHamtyTUk9LS1zTHJHSDJoOE12cnlmZU0vbEhKN3R3PT0%3D--03003ee363cfd2faf6e5cf2e8532e7c3a89001a5; socrata-csrf-token=UJYT0IG17E5dIJRGRcstFuUrxTR62o7pITdYYW8r5vjn2sWpSipAmGuDqHctZcFbAtG75rh3AYZsqbbUceblTA",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
        "Priority": "u=0"
    }
    
    payload = {
        "query": "SELECT\n  `crash_record_id`,\n  `crash_date_est_i`,\n  `crash_date`,\n  `posted_speed_limit`,\n  `traffic_control_device`,\n  `device_condition`,\n  `weather_condition`,\n  `lighting_condition`,\n  `first_crash_type`,\n  `trafficway_type`,\n  `lane_cnt`,\n  `alignment`,\n  `roadway_surface_cond`,\n  `road_defect`,\n  `report_type`,\n  `crash_type`,\n  `intersection_related_i`,\n  `private_property_i`,\n  `hit_and_run_i`,\n  `damage`,\n  `date_police_notified`,\n  `prim_contributory_cause`,\n  `sec_contributory_cause`,\n  `street_no`,\n  `street_direction`,\n  `street_name`,\n  `beat_of_occurrence`,\n  `photos_taken_i`,\n  `statements_taken_i`,\n  `dooring_i`,\n  `work_zone_i`,\n  `work_zone_type`,\n  `workers_present_i`,\n  `num_units`,\n  `most_severe_injury`,\n  `injuries_total`,\n  `injuries_fatal`,\n  `injuries_incapacitating`,\n  `injuries_non_incapacitating`,\n  `injuries_reported_not_evident`,\n  `injuries_no_indication`,\n  `injuries_unknown`,\n  `crash_hour`,\n  `crash_day_of_week`,\n  `crash_month`,\n  `latitude`,\n  `longitude`,\n  `location`\nWHERE\n  `crash_date`\n    BETWEEN \"2024-01-01T11:07:33\" :: floating_timestamp\n    AND \"2024-01-30T11:07:33\" :: floating_timestamp\nORDER BY `crash_date` DESC NULL FIRST, `crash_record_id` ASC NULL LAST",
        "serializationOptions": {
            "defaultGroupSeparator": ",",
            "defaultDecimalSeparator": "."
        }
    }
    
    response = requests.post(url, headers=headers, json=payload, stream=True)
    response.raise_for_status()
    
    with open(f"data/{filename}", "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)