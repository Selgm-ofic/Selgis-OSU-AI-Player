using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using System.Text.Json;
using System.Diagnostics;
using Realms;

class Program
{
    static void Main(string[] args)
    {
        var sw = Stopwatch.StartNew();
        string realmPath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData), "osu", "client.realm");
        string outputPath = args.Length > 0 ? args[0] : Path.Combine(AppContext.BaseDirectory, "lazer_manifest.json");

        if (!File.Exists(realmPath))
        {
            Console.Error.WriteLine($"Realm file not found at: {realmPath}");
            Environment.Exit(1);
        }

        var config = new RealmConfiguration(realmPath)
        {
            IsReadOnly = true,
            IsDynamic = true
        };

        var realm = Realm.GetInstance(config);
        var sets = realm.DynamicApi.All("BeatmapSet").ToArray();
        Console.WriteLine($"Total BeatmapSets in Realm: {sets.Length}");

        var manifestSets = new List<object>();
        int totalStandardMaps = 0;

        foreach (var set in sets)
        {
            var filesList = set.DynamicApi.GetList<IRealmObjectBase>("Files");
            var bmsList = set.DynamicApi.GetList<IRealmObjectBase>("Beatmaps");

            string title = "Unknown";
            string artist = "Unknown";
            string creator = "Unknown";
            long setId = -1;
            try { setId = set.DynamicApi.Get<long>("OnlineID"); } catch {}
            string guidStr = "";
            try { guidStr = set.DynamicApi.Get<Guid>("ID").ToString(); } catch {}

            var beatmaps = new List<object>();
            foreach (var b in bmsList)
            {
                int ruleset = 0;
                try
                {
                    var rs = b.DynamicApi.Get<IRealmObjectBase>("Ruleset");
                    if (rs != null) ruleset = (int)rs.DynamicApi.Get<long>("OnlineID");
                }
                catch {}

                // Only osu! standard (ruleset 0)
                if (ruleset != 0) continue;

                string diff = b.DynamicApi.Get<string>("DifficultyName") ?? "Normal";
                double star = 0;
                try { star = b.DynamicApi.Get<double>("StarRating"); } catch {}
                long mapOnlineId = -1;
                try { mapOnlineId = b.DynamicApi.Get<long>("OnlineID"); } catch {}
                string mapGuid = "";
                try { mapGuid = b.DynamicApi.Get<Guid>("ID").ToString(); } catch {}
                string hash = "";
                try { hash = b.DynamicApi.Get<string>("MD5Hash") ?? ""; } catch {}

                // Get metadata from beatmap if available
                try
                {
                    var meta = b.DynamicApi.Get<IRealmObjectBase>("Metadata");
                    if (meta != null)
                    {
                        if (title == "Unknown") title = meta.DynamicApi.Get<string>("Title") ?? title;
                        if (artist == "Unknown") artist = meta.DynamicApi.Get<string>("Artist") ?? artist;
                        if (creator == "Unknown") creator = meta.DynamicApi.Get<string>("Author") ?? meta.DynamicApi.Get<string>("AuthorString") ?? creator;
                    }
                }
                catch {}

                beatmaps.Add(new
                {
                    difficulty = diff,
                    star_rating = Math.Round(star, 2),
                    online_id = mapOnlineId,
                    guid = mapGuid,
                    md5 = hash
                });
                totalStandardMaps++;
            }

            if (beatmaps.Count == 0) continue; // No osu! standard beatmaps in this set

            var files = new List<object>();
            foreach (var f in filesList)
            {
                string filename = f.DynamicApi.Get<string>("Filename");
                var fileObj = f.DynamicApi.Get<IRealmObjectBase>("File");
                string fileHash = fileObj != null ? fileObj.DynamicApi.Get<string>("Hash") : null;
                if (!string.IsNullOrEmpty(filename) && !string.IsNullOrEmpty(fileHash))
                {
                    files.Add(new
                    {
                        filename = filename,
                        hash = fileHash
                    });
                }
            }

            // Clean folder name
            string folderPrefix = setId > 0 ? setId.ToString() : guidStr.Substring(0, 8);
            string safeArtist = string.Join("_", artist.Split(Path.GetInvalidFileNameChars())).Trim();
            string safeTitle = string.Join("_", title.Split(Path.GetInvalidFileNameChars())).Trim();
            string folderName = $"{folderPrefix} {safeArtist} - {safeTitle}".Trim();

            manifestSets.Add(new
            {
                guid = guidStr,
                online_id = setId,
                artist = artist,
                title = title,
                creator = creator,
                folder_name = folderName,
                beatmaps = beatmaps,
                files = files
            });
        }

        var result = new
        {
            extracted_at = DateTime.UtcNow.ToString("o"),
            total_sets = manifestSets.Count,
            total_standard_beatmaps = totalStandardMaps,
            sets = manifestSets
        };

        var jsonOptions = new JsonSerializerOptions { WriteIndented = true };
        string json = JsonSerializer.Serialize(result, jsonOptions);
        File.WriteAllText(outputPath, json);

        sw.Stop();
        Console.WriteLine($"Extracted {manifestSets.Count} sets ({totalStandardMaps} osu! standard beatmaps) to {outputPath} in {sw.ElapsedMilliseconds} ms.");
    }
}

