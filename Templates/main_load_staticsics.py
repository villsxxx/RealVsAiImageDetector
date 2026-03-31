import pandas as pd
from tensorboard.backend.event_processing import event_accumulator


def extract_epoch_metrics(logdir):
    ea = event_accumulator.EventAccumulator(logdir)
    ea.Reload()

    tags = ea.Tags()['scalars']
    data = {}

    for tag in tags:
        events = ea.Scalars(tag)

        steps = []
        values = []
        current_epoch = -1

        for event in events:
            step = event.step
            value = event.value
            epoch = step // 927 if 'train' in tag else step // 31

            if epoch > current_epoch:
                if current_epoch >= 0:
                    steps.append(current_epoch)
                    values.append(last_value)
                current_epoch = epoch
                last_value = value
            else:
                last_value = value

        if current_epoch >= 0:
            steps.append(current_epoch)
            values.append(last_value)

        data[tag] = pd.DataFrame({'epoch': steps, 'value': values})

    return data


log_dirs = ["D:/nnModels\RGDetector\ArtClassification/0326_001816\models/tensorboard/version_0"]
if log_dirs:
    print(f"Found logs: {log_dirs[0]}")
    data = extract_epoch_metrics(log_dirs[0])

    for tag, df in data.items():
        print(f"\n{tag}:")
        print(df.to_string(index=False))

        df.to_csv(f"{tag.replace('/', '_')}_per_epoch.csv", index=False)